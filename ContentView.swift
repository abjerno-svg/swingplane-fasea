//
//  ContentView.swift
//  SwingPlane – Fase A (kamera + pose + framing + bibliotek + auto-optagelse + sync-review)
//
//  ALT i én fil (pipeline-fil). Build 5-batch:
//   - Auto-optagelse på impact-lyd: arm → sving → auto-trimmet klip (rolling-trim)
//   - Cached pose (sidefil) → ALTID-PÅ skelet i afspilleren (ingen knap)
//   - Side-om-side synkron DTL/Face-on-review, justeret på impact
//   + tidligere: front+flip, høj-fps, live pose, framing-gates, lyd-meter, app-privat lager, bibliotek
//
//  TUNING-PUNKTER (device): impact-tærskel + trim-vindue, sync-lead, framing-tærskler, pose-orientering.
//

import SwiftUI
import Combine
import AVFoundation
import AVKit
import UIKit
import Vision
import CoreMotion
import CoreML

typealias PoseDict = [VNHumanBodyPoseObservation.JointName: CGPoint]

enum SwingView: String, CaseIterable {
    case faceOn = "Face-on"
    case dtl = "DTL"
    var fileKey: String { self == .dtl ? "DTL" : "FaceOn" }
    static func fromFileKey(_ s: String) -> SwingView { s == "DTL" ? .dtl : .faceOn }
}

// MARK: - Pose-cache (Codable sidefil)

struct PoseFrame: Codable { let t: Double; let joints: [String: [Double]] }
struct PoseCache: Codable { let width: Double; let height: Double; let frames: [PoseFrame] }
struct ClubFrame: Codable { let t: Double; let x: Double; let y: Double; let c: Double }
struct ClubCache: Codable {
    let v: Int; let width: Double; let height: Double; let frames: [ClubFrame]
    var ballCands: [[Double]] = []    // [x,y,conf] bold-kandidater i klippets foerste 0,6s
}

enum PoseKeys {
    static let map: [(VNHumanBodyPoseObservation.JointName, String)] = [
        (.nose, "nose"), (.neck, "neck"),
        (.leftShoulder, "lSho"), (.rightShoulder, "rSho"),
        (.leftElbow, "lElb"), (.rightElbow, "rElb"),
        (.leftWrist, "lWri"), (.rightWrist, "rWri"),
        (.leftHip, "lHip"), (.rightHip, "rHip"),
        (.leftKnee, "lKne"), (.rightKnee, "rKne"),
        (.leftAnkle, "lAnk"), (.rightAnkle, "rAnk")
    ]
    static let toKey = Dictionary(uniqueKeysWithValues: map.map { ($0.0, $0.1) })
    static let toJoint = Dictionary(uniqueKeysWithValues: map.map { ($0.1, $0.0) })
}

// MARK: - Klip-model + lager

// MARK: - Klubhoved-analyse (Core ML, en gang pr. klip, cache som sidefil)
// Kraever SwingClub.mlpackage i app-bundlet (konverteret fra best_v3.onnx).
// Mangler modellen, degraderer den paent: ingen trail, ingen crash.

enum ClubAnalyzer {
    static func cacheURL(for clipURL: URL) -> URL {
        clipURL.appendingPathExtension("club.json")
    }

    static var model: MLModel? = {
        guard let url = Bundle.main.url(forResource: "SwingClub", withExtension: "mlmodelc") else { return nil }
        let cfg = MLModelConfiguration(); cfg.computeUnits = .all
        return try? MLModel(contentsOf: url, configuration: cfg)
    }()

    static func loadOrCompute(for clipURL: URL, completion: @escaping (ClubCache?) -> Void) {
        let cacheURL = cacheURL(for: clipURL)
        if let data = try? Data(contentsOf: cacheURL),
           let cache = try? JSONDecoder().decode(ClubCache.self, from: data), cache.v >= 7 {
            completion(cache); return
        }
        guard let model = model else { completion(nil); return }
        DispatchQueue.global(qos: .userInitiated).async {
            let asset = AVURLAsset(url: clipURL)
            let dur = CMTimeGetSeconds(asset.duration)
            let gen = AVAssetImageGenerator(asset: asset)
            gen.appliesPreferredTrackTransform = true
            gen.requestedTimeToleranceBefore = .zero
            gen.requestedTimeToleranceAfter = .zero
            gen.maximumSize = CGSize(width: 640, height: 640)   // modellen ser 640 — fuld 1080x1920 = 9x spildt RAM (jetsam)
            var frames: [ClubFrame] = []
            var ballCands: [[Double]] = []
            var w = 1080.0, h = 1920.0
            let step = 1.0 / 60.0     // taettere end pose: trail skal vaere glat
            var t = 0.0
            while t < max(dur, 0.01) {
                autoreleasepool {
                    if let cg = try? gen.copyCGImage(at: CMTime(seconds: t, preferredTimescale: 600), actualTime: nil) {
                        w = Double(cg.width); h = Double(cg.height)
                        let det = detect(cg: cg, model: model)
                        if let c = det.club {
                            frames.append(ClubFrame(t: t, x: c.x, y: c.y, c: c.c))
                        }
                        if t <= 0.6 {
                            for b in det.balls { ballCands.append([b.x, b.y, b.c]) }
                        }
                    }
                }
                t += step
            }
            var cache = ClubCache(v: 7, width: w, height: h, frames: frames)
            cache.ballCands = ballCands
            if let data = try? JSONEncoder().encode(cache) { try? data.write(to: cacheURL) }
            DispatchQueue.main.async { completion(cache) }
        }
    }

    /// Letterbox 640x640 (top-venstre, praecis som valideret sandbox-pipeline), koer modellen.
    /// v3-klasser: raekke 4 = bold, raekke 5 = KLUBHOVED, raekke 6 = haender.
    /// Stride-sikker aflaesning (ANE kan levere ikke-kontinuert MLMultiArray).
    private static func detect(cg: CGImage, model: MLModel)
        -> (club: (x: Double, y: Double, c: Double)?, balls: [(x: Double, y: Double, c: Double)]) {
        let W = cg.width, H = cg.height
        let sc = 640.0 / Double(max(W, H))
        let sw = Int(Double(W) * sc), sh = Int(Double(H) * sc)
        var pb: CVPixelBuffer?
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: true,
                     kCVPixelBufferCGBitmapContextCompatibilityKey: true] as CFDictionary
        CVPixelBufferCreate(kCFAllocatorDefault, 640, 640, kCVPixelFormatType_32BGRA, attrs, &pb)
        guard let buf = pb else { return (nil, []) }
        CVPixelBufferLockBaseAddress(buf, [])
        defer { CVPixelBufferUnlockBaseAddress(buf, []) }
        guard let ctx = CGContext(data: CVPixelBufferGetBaseAddress(buf),
                                  width: 640, height: 640, bitsPerComponent: 8,
                                  bytesPerRow: CVPixelBufferGetBytesPerRow(buf),
                                  space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                                      | CGBitmapInfo.byteOrder32Little.rawValue) else { return (nil, []) }
        ctx.setFillColor(CGColor(red: 114/255, green: 114/255, blue: 114/255, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: 640, height: 640))
        // Ingen flip: CGContext.draw laegger CGImage OPRET i bufferen.
        // Rect oeverst i CG-koordinater (y: 640-sh) = raekke 0..sh i hukommelsen = TOP-venstre.
        ctx.draw(cg, in: CGRect(x: 0, y: 640 - sh, width: sw, height: sh))
        // Navne-agnostisk: v3-konverteringen hed images/output0, ultralytics-eksport hedder image/var_xxx
        guard let inputName = model.modelDescription.inputDescriptionsByName.keys.first,
              let inp = try? MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(pixelBuffer: buf)]),
              let outp = try? model.prediction(from: inp) else { return (nil, []) }
        var arrOpt: MLMultiArray? = nil
        for name in outp.featureNames {
            if let a = outp.featureValue(for: name)?.multiArrayValue,
               a.shape.map(\.intValue).contains(8400) { arrOpt = a; break }
        }
        guard let arr = arrOpt else { return (nil, []) }
        let n = 8400
        let shape = arr.shape.map(\.intValue)
        let strides = arr.strides.map(\.intValue)
        var chS = n, iS = 1, nCh = 7
        if shape.count == 3 { chS = strides[1]; iS = strides[2]; nCh = shape[1] }
        else if shape.count == 2 { chS = strides[0]; iS = strides[1]; nCh = shape[0] }
        guard nCh >= 7, shape.last == n || shape.count == 3 else { return (nil, []) }
        let base = arr.dataPointer
        let dt = arr.dataType
        func row(_ ch: Int) -> [Float] {
            var out = [Float](repeating: 0, count: n)
            let maxIdx = ch * chS + (n - 1) * iS + 1
            switch dt {
            case .float32:
                let p = base.bindMemory(to: Float.self, capacity: maxIdx)
                for i in 0..<n { out[i] = p[ch * chS + i * iS] }
            case .float16:
                let p = base.bindMemory(to: Float16.self, capacity: maxIdx)
                for i in 0..<n { out[i] = Float(p[ch * chS + i * iS]) }
            case .double:
                let p = base.bindMemory(to: Double.self, capacity: maxIdx)
                for i in 0..<n { out[i] = Float(p[ch * chS + i * iS]) }
            default: break
            }
            return out
        }
        let cxr = row(0), cyr = row(1)
        func best(_ conf: [Float], _ thr: Float) -> (x: Double, y: Double, c: Double)? {
            var bC: Float = 0; var bI = -1
            for i in 0..<n where conf[i] > bC { bC = conf[i]; bI = i }
            guard bC > thr, bI >= 0 else { return nil }
            return (x: (Double(cxr[bI]) / sc) / Double(W),
                    y: (Double(cyr[bI]) / sc) / Double(H),
                    c: Double(bC))
        }
        let ballConf = row(4)
        var balls: [(x: Double, y: Double, c: Double)] = []
        var used: [(Double, Double)] = []
        for _ in 0..<3 {
            var bC: Float = 0; var bI = -1
            for i in 0..<n where ballConf[i] > bC {
                let x = Double(cxr[i]), y = Double(cyr[i])
                if used.contains(where: { abs($0.0 - x) < 15 && abs($0.1 - y) < 15 }) { continue }
                bC = ballConf[i]; bI = i
            }
            guard bC > 0.12, bI >= 0 else { break }
            used.append((Double(cxr[bI]), Double(cyr[bI])))
            balls.append((x: (Double(cxr[bI]) / sc) / Double(W),
                          y: (Double(cyr[bI]) / sc) / Double(H), c: Double(bC)))
        }
        return (club: best(row(5), 0.25), balls: balls)
    }

    /// Split-indeks til farveskift: toppen = hoejeste klubhoved-punkt FOER impact.
    static func topIndex(in cache: ClubCache, impact: Double?) -> Int {
        let cut = impact ?? (cache.frames.last?.t ?? 0) * 0.6
        var best = 0; var bestY = 2.0
        for (i, f) in cache.frames.enumerated() where f.t <= cut {
            if f.y < bestY { bestY = f.y; best = i }
        }
        return best
    }
}

// MARK: - Baggrunds-analyse (pose + klubhoved koeres straks naar et klip er gemt,
// saa toggles som regel er klar naar man aabner klippet). Serielt, nyeste foerst.

enum BackgroundAnalyzer {
    private static var running = false
    private static var pending: [URL] = []

    /// Kaldes fra ClipStore.reload() paa main. Saetter koen = klip der mangler cache.
    static func ensure(_ clips: [Clip]) {
        let fm = FileManager.default
        pending = clips.map(\.url).filter { url in
            !fm.fileExists(atPath: PoseAnalyzer.cacheURL(for: url).path)
                || !fm.fileExists(atPath: ClubAnalyzer.cacheURL(for: url).path)
        }
        kick()
    }

    private static func kick() {
        guard !running, !pending.isEmpty else { return }
        running = true
        let url = pending.removeFirst()
        PoseAnalyzer.loadOrCompute(for: url) { _ in
            ClubAnalyzer.loadOrCompute(for: url) { _ in
                running = false
                kick()
            }
        }
    }
}

struct Clip: Identifiable {
    let id = UUID()
    let url: URL
    let view: SwingView
    let date: Date
    let impact: Double?     // sekunder inde i klippet, hvis kendt
    let number: Int         // stabilt sving-nummer (aendres ALDRIG ved sletning)
    let comment: String
}

/// Sidefil .meta.json: stabilt nummer + brugerens kommentar
struct ClipMeta: Codable { var n: Int; var comment: String? }

final class ClipStore: ObservableObject {
    @Published var clips: [Clip] = []

    static let directory: URL = {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let dir = docs.appendingPathComponent("clips", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }()

    static let maxClips = 80     // auto-loft: behold nyeste N, slet ældste

    /// Filnavn: swing_<ts>_<viewKey>_<impactMs|na>.mov
    static func newURL(view: SwingView, impactMs: Int?) -> URL {
        let ts = Int(Date().timeIntervalSince1970)
        let imp = impactMs.map(String.init) ?? "na"
        return directory.appendingPathComponent("swing_\(ts)_\(view.fileKey)_\(imp).mov")
    }

    static func metaURL(for url: URL) -> URL {
        url.deletingPathExtension().appendingPathExtension("meta.json")
    }

    func setComment(_ clip: Clip, _ text: String) {
        var meta = ClipMeta(n: clip.number, comment: text)
        if let d = try? Data(contentsOf: Self.metaURL(for: clip.url)),
           let m = try? JSONDecoder().decode(ClipMeta.self, from: d) { meta = m; meta.comment = text }
        if let d = try? JSONEncoder().encode(meta) { try? d.write(to: Self.metaURL(for: clip.url)) }
        if let i = clips.firstIndex(where: { $0.id == clip.id }) {
            clips[i] = Clip(url: clip.url, view: clip.view, date: clip.date, impact: clip.impact,
                            number: clip.number, comment: text)
        }
    }

    func reload() {
        let files = (try? FileManager.default.contentsOfDirectory(
            at: Self.directory, includingPropertiesForKeys: nil)) ?? []
        var parsed: [(url: URL, view: SwingView, date: Date, impact: Double?)] = []
        for url in files where url.pathExtension == "mov" {
            let parts = url.deletingPathExtension().lastPathComponent.split(separator: "_")
            guard parts.count >= 3, let ts = Double(parts[1]) else { continue }
            let view = SwingView.fromFileKey(String(parts[2]))
            var impact: Double? = nil
            if parts.count >= 4, let ms = Double(parts[3]) { impact = ms / 1000 }
            parsed.append((url, view, Date(timeIntervalSince1970: ts), impact))
        }
        // Stabile sving-numre: .meta.json pr. klip + monoton taeller i UserDefaults
        var counter = UserDefaults.standard.integer(forKey: "swingCounter")
        var result: [Clip] = []
        for e in parsed.sorted(by: { $0.date < $1.date }) {
            let mURL = Self.metaURL(for: e.url)
            let meta: ClipMeta
            if let d = try? Data(contentsOf: mURL), let m = try? JSONDecoder().decode(ClipMeta.self, from: d) {
                meta = m
                counter = max(counter, m.n)
            } else {
                counter += 1
                meta = ClipMeta(n: counter, comment: nil)
                if let d = try? JSONEncoder().encode(meta) { try? d.write(to: mURL) }
            }
            result.append(Clip(url: e.url, view: e.view, date: e.date, impact: e.impact,
                               number: meta.n, comment: meta.comment ?? ""))
        }
        UserDefaults.standard.set(counter, forKey: "swingCounter")
        let sorted = result.sorted { $0.date > $1.date }
        // Auto-loft: slet ældste ud over maxClips
        if sorted.count > Self.maxClips {
            for c in sorted[Self.maxClips...] { removeFiles(c) }
            clips = Array(sorted.prefix(Self.maxClips))
        } else {
            clips = sorted
        }
        BackgroundAnalyzer.ensure(clips)
    }

    private func removeFiles(_ clip: Clip) {
        try? FileManager.default.removeItem(at: clip.url)
        try? FileManager.default.removeItem(at: PoseAnalyzer.cacheURL(for: clip.url))
        try? FileManager.default.removeItem(at: ClubAnalyzer.cacheURL(for: clip.url))
        try? FileManager.default.removeItem(at: clip.url.deletingPathExtension().appendingPathExtension("thumb.jpg"))
        try? FileManager.default.removeItem(at: Self.metaURL(for: clip.url))
    }

    func delete(_ clip: Clip) { removeFiles(clip); reload() }

    func deleteAll() {
        for c in clips { removeFiles(c) }
        reload()
    }
}

// MARK: - Pose-analyse (cache-beregning)

enum PoseAnalyzer {
    static func cacheURL(for clipURL: URL) -> URL {
        clipURL.appendingPathExtension("pose.json")
    }

    /// Henter cache fra sidefil, eller beregner (Vision på samplede frames) + gemmer.
    static func loadOrCompute(for clipURL: URL, completion: @escaping (PoseCache) -> Void) {
        let cacheURL = cacheURL(for: clipURL)
        if let data = try? Data(contentsOf: cacheURL),
           let cache = try? JSONDecoder().decode(PoseCache.self, from: data) {
            completion(cache); return
        }
        DispatchQueue.global(qos: .userInitiated).async {
            let asset = AVURLAsset(url: clipURL)
            let dur = CMTimeGetSeconds(asset.duration)
            let gen = AVAssetImageGenerator(asset: asset)
            gen.appliesPreferredTrackTransform = true
            gen.requestedTimeToleranceBefore = .zero
            gen.requestedTimeToleranceAfter = .zero
            gen.maximumSize = CGSize(width: 1024, height: 1024)  // jetsam-fix: Vision behoever ikke 1080x1920

            var frames: [PoseFrame] = []
            var w = 1080.0, h = 1920.0
            let step = 1.0 / 30.0
            var t = 0.0
            while t < max(dur, 0.01) {
                autoreleasepool { () -> Void in
                if let cg = try? gen.copyCGImage(at: CMTime(seconds: t, preferredTimescale: 600), actualTime: nil) {
                    w = Double(cg.width); h = Double(cg.height)
                    let handler = VNImageRequestHandler(cgImage: cg, orientation: .up, options: [:])
                    let req = VNDetectHumanBodyPoseRequest()
                    try? handler.perform([req])
                    var joints: [String: [Double]] = [:]
                    if let obs = req.results?.first, let pts = try? obs.recognizedPoints(.all) {
                        for (name, p) in pts where p.confidence > 0.3 {
                            if let key = PoseKeys.toKey[name] {
                                joints[key] = [Double(p.location.x), Double(1 - p.location.y)]
                            }
                        }
                    }
                    frames.append(PoseFrame(t: t, joints: joints))
                }
                }
                t += step
            }
            let cache = PoseCache(width: w, height: h, frames: frames)
            if let data = try? JSONEncoder().encode(cache) { try? data.write(to: cacheURL) }
            DispatchQueue.main.async { completion(cache) }
        }
    }

    static func pose(at time: Double, in cache: PoseCache) -> PoseDict {
        guard let frame = cache.frames.min(by: { abs($0.t - time) < abs($1.t - time) }) else { return [:] }
        var result: PoseDict = [:]
        for (key, xy) in frame.joints {
            if let joint = PoseKeys.toJoint[key], xy.count == 2 {
                result[joint] = CGPoint(x: xy[0], y: xy[1])
            }
        }
        return result
    }
}

// MARK: - Rod-view (kamera + bibliotek-sheet)

struct CameraScreen: View {
    var mode: SessionMode = .multiple
    var angle: SwingView = .dtl
    var onExit: () -> Void = {}
    @State private var focusArea: FocusArea = .impact
    @State private var trainingMode: TrainingMode = .driver
    @StateObject private var camera = CameraManager()
    @StateObject private var store = ClipStore()
    @State private var showLibrary = false

    var body: some View {
        ZStack {
            if camera.permissionGranted {
                CameraPreview(session: camera.session).ignoresSafeArea()
            } else {
                Color.black.ignoresSafeArea()
            }

            PoseOverlay(pose: camera.pose, videoSize: camera.videoPortraitSize)
                .allowsHitTesting(false).ignoresSafeArea()

            FramingBox(inside: camera.allChecksPass, hint: camera.frameHint)
                .allowsHitTesting(false)

            HStack {
                AudioMeter(level: camera.audioLevel, impact: camera.impactFlash)
                    .frame(width: 8).padding(.leading, 6)
                Spacer()
            }.ignoresSafeArea(edges: .vertical)

            VStack(spacing: 8) {
                HStack {
                    Button { camera.pauseListening(); showLibrary = true } label: {
                        Label("My swings", systemImage: "square.stack.3d.up.fill")
                            .font(.system(size: 16, weight: .semibold)).foregroundStyle(.white)
                            .padding(.horizontal, 16).padding(.vertical, 12)
                            .background(.black.opacity(0.55)).clipShape(Capsule())
                    }
                    Spacer()
                    Menu {
                        Picker("Fokus", selection: $focusArea) {
                            ForEach(FocusArea.allCases) { Text($0.rawValue).tag($0) }
                        }
                        Picker("Træningsmode", selection: $trainingMode) {
                            ForEach(TrainingMode.allCases) { Text($0.rawValue).tag($0) }
                        }
                        Picker("Følsomhed (blæst)", selection: $camera.windSensitivity) {
                            ForEach(WindSensitivity.allCases) { Text($0.rawValue).tag($0) }
                        }
                        Toggle(isOn: $camera.poseGateEnabled) {
                            Label("Kun ved sving (pose-gate)", systemImage: "figure.golf")
                        }
                        Button { camera.flipCamera() } label: {
                            Label("Skift kamera", systemImage: "camera.rotate")
                        }
                        Divider()
                        Button { onExit() } label: {
                            Label("Start forfra", systemImage: "arrow.uturn.left")
                        }
                    } label: {
                        Image(systemName: "line.3.horizontal")
                            .font(.system(size: 20, weight: .semibold)).foregroundStyle(.white)
                            .padding(12).background(.black.opacity(0.55)).clipShape(Circle())
                    }
                }
                HStack {
                    Spacer()
                    Toggle(isOn: $camera.autoMode) {
                        Label("Range session", systemImage: "flag.fill").font(.caption.weight(.bold))
                    }
                    .toggleStyle(.button).tint(.spGold)
                    .buttonBorderShape(.capsule)
                    .overlay(Capsule().stroke(Color.spGold, lineWidth: 1.5))
                }
                .padding(.top, 10)
                if camera.autoListening { ListeningBadge() }
                // Valideringer i venstre side - ikke midt i billedet (front-kamera)
                HStack {
                    VStack(alignment: .leading, spacing: 6) {
                        badge(camera.statusText)
                        badge(camera.poseInfo)
                        FramingChecklist(camera: camera)
                    }
                    Spacer()
                }
                .padding(.top, 4)

                Spacer()

                AngleChips(selected: $camera.selectedView).padding(.bottom, 6)

                ZStack {
                    Button(action: { camera.toggleRecording() }) {
                        ZStack {
                            Circle().stroke(.white, lineWidth: 4).frame(width: 84, height: 84)
                            Circle().fill(camera.isRecording ? Color.red : Color.white)
                                .frame(width: 70, height: 70)
                        }
                    }.disabled(!camera.permissionGranted)

                    HStack {
                        Spacer()
                        Button(action: { camera.flipCamera() }) {
                            Image(systemName: "arrow.triangle.2.circlepath.camera")
                                .font(.system(size: 24, weight: .semibold)).foregroundStyle(.white)
                                .padding(16).background(.black.opacity(0.55)).clipShape(Circle())
                        }
                        .disabled(!camera.permissionGranted)
                        .padding(.trailing, 32)
                    }
                }.padding(.bottom, 40)
            }
            .padding([.horizontal, .top], 12)
        }
        .onAppear {
            camera.selectedView = angle
            camera.autoMode = (mode == .multiple)
            camera.onClipSaved = { store.reload() }
            camera.start(); store.reload()
        }
        .sheet(isPresented: $showLibrary, onDismiss: { camera.resumeListening() }) { LibraryView(store: store) }
    }

    private func badge(_ text: String) -> some View {
        Text(text).font(.caption.weight(.semibold))
            .padding(.horizontal, 12).padding(.vertical, 6)
            .background(.black.opacity(0.55)).foregroundStyle(.white).clipShape(Capsule())
    }
}

#Preview { ContentView() }

// MARK: - Vinkel-chips + lytter-badge

struct AngleChips: View {
    @Binding var selected: SwingView
    var body: some View {
        HStack(spacing: 10) {
            ForEach(SwingView.allCases, id: \.self) { v in
                Button { selected = v } label: {
                    Text(v.rawValue)
                        .font(.subheadline.weight(.bold))
                        .padding(.horizontal, 18).padding(.vertical, 9)
                        .background(selected == v ? Color.spGold : Color.black.opacity(0.5))
                        .foregroundStyle(selected == v ? Color.spInk : .white).clipShape(Capsule())
                        .overlay(Capsule().stroke(selected == v ? Color.spGoldLight : .clear, lineWidth: 1.5))
                }
            }
        }
    }
}

struct ListeningBadge: View {
    @State private var pulse = false
    var body: some View {
        HStack(spacing: 8) {
            Circle().fill(.red).frame(width: 10, height: 10)
                .scaleEffect(pulse ? 1.4 : 0.8).opacity(pulse ? 1 : 0.5)
            Text("LYTTER – slå dit sving").font(.caption.weight(.bold)).foregroundStyle(.white)
        }
        .padding(.horizontal, 14).padding(.vertical, 8)
        .background(.red.opacity(0.35)).clipShape(Capsule())
        .overlay(Capsule().stroke(.red, lineWidth: 1))
        .onAppear { withAnimation(.easeInOut(duration: 0.7).repeatForever(autoreverses: true)) { pulse = true } }
    }
}

// MARK: - Bibliotek

struct LibraryView: View {
    @ObservedObject var store: ClipStore
    @Environment(\.dismiss) private var dismiss
    @State private var filter: SwingView? = nil
    @State private var showSync = false
    @State private var confirmDeleteAll = false

    private var filtered: [Clip] {
        guard let f = filter else { return store.clips }
        return store.clips.filter { $0.view == f }
    }

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                Picker("Filter", selection: $filter) {
                    Text("Begge").tag(nil as SwingView?)
                    Text("Face-on").tag(SwingView.faceOn as SwingView?)
                    Text("DTL").tag(SwingView.dtl as SwingView?)
                }
                .pickerStyle(.segmented).padding()

                HStack {
                    Button { showSync = true } label: {
                        Label("Sammenlign", systemImage: "rectangle.split.2x1")
                    }
                    Spacer()
                    Button(role: .destructive) { confirmDeleteAll = true } label: {
                        Label("Slet alle", systemImage: "trash")
                    }
                    .disabled(store.clips.isEmpty)
                }
                .padding(.horizontal).padding(.bottom, 8)

                if filtered.isEmpty {
                    Spacer(); Text("Ingen sving endnu").foregroundStyle(.secondary); Spacer()
                } else {
                    List {
                        ForEach(filtered) { clip in
                            NavigationLink { ClipPlayerView(clip: clip, store: store) } label: { ClipRow(clip: clip) }
                        }
                        .onDelete { offsets in offsets.forEach { store.delete(filtered[$0]) } }
                    }
                }
            }
            .navigationTitle("Seneste sving")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Luk") { dismiss() } }
                ToolbarItem(placement: .primaryAction) { EditButton() }
            }
            .confirmationDialog("Slet alle sving?", isPresented: $confirmDeleteAll, titleVisibility: .visible) {
                Button("Slet alle", role: .destructive) { store.deleteAll() }
                Button("Annuller", role: .cancel) {}
            }
            .sheet(isPresented: $showSync) { SyncReviewView(store: store) }
        }
        .onAppear { store.reload() }
    }
}

struct ClipRow: View {
    let clip: Clip
    @State private var durationText = ""
    private static let fmt: DateFormatter = {
        let f = DateFormatter(); f.dateFormat = "d. MMM HH:mm"; return f
    }()
    static func dateString(_ d: Date) -> String { fmt.string(from: d) }
    var body: some View {
        HStack(spacing: 12) {
            ClipThumbnail(url: clip.url)
            VStack(alignment: .leading, spacing: 3) {
                Text("Sving \(clip.number)").font(.body.weight(.semibold))
                HStack(spacing: 6) {
                    Image(systemName: clip.view == .dtl ? "figure.golf" : "person.fill")
                        .font(.caption2).foregroundStyle(Color.spGold)
                    Text(clip.view.rawValue).font(.caption).foregroundStyle(.secondary)
                    if !durationText.isEmpty {
                        Text("· \(durationText)").font(.caption).foregroundStyle(.secondary)
                    }
                }
                Text(Self.fmt.string(from: clip.date)).font(.caption2).foregroundStyle(.secondary)
                if !clip.comment.isEmpty {
                    HStack(spacing: 4) {
                        Image(systemName: "text.bubble").font(.caption2).foregroundStyle(Color.spGold)
                        Text(clip.comment).font(.caption2).foregroundStyle(.secondary).lineLimit(1)
                    }
                }
            }
            Spacer()
            Image(systemName: "play.circle").foregroundStyle(.secondary)
        }
        .padding(.vertical, 2)
        .task {
            guard durationText.isEmpty else { return }
            let sec = (try? await AVURLAsset(url: clip.url).load(.duration).seconds) ?? 0
            if sec.isFinite && sec > 0 { durationText = String(format: "%.1f s", sec) }
        }
    }
}

struct ClipThumbnail: View {
    let url: URL
    @State private var img: UIImage?
    var body: some View {
        ZStack {
            if let img {
                Image(uiImage: img).resizable().scaledToFill()
            } else {
                Color.white.opacity(0.08)
                Image(systemName: "figure.golf").foregroundStyle(.secondary)
            }
        }
        .frame(width: 52, height: 70)
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .task { await load() }
    }
    private func load() async {
        let thumb = url.deletingPathExtension().appendingPathExtension("thumb.jpg")
        if let data = try? Data(contentsOf: thumb), let ui = UIImage(data: data) { img = ui; return }
        let asset = AVURLAsset(url: url)
        let gen = AVAssetImageGenerator(asset: asset)
        gen.appliesPreferredTrackTransform = true
        gen.maximumSize = CGSize(width: 220, height: 320)
        let t = CMTime(seconds: 0.1, preferredTimescale: 600)
        if let cg = try? await gen.image(at: t).image {
            let ui = UIImage(cgImage: cg)
            img = ui
            if let jpg = ui.jpegData(compressionQuality: 0.7) { try? jpg.write(to: thumb) }
        }
    }
}

// MARK: - Skelet-video (genbrugelig: video + altid-på skelet fra cache)

struct SkeletonVideo: View {
    let clip: Clip
    let player: AVPlayer

    @State private var cache: PoseCache?
    @State private var currentPose: PoseDict = [:]
    @State private var analyzing = true
    @State private var observer: Any?
    @State private var clubCache: ClubCache?
    @State private var clubAnalyzing = true
    @State private var clubTopIdx = 0
    @State private var currentTime: Double = 0
    @State private var planeLines: [PlaneLine] = []
    var showClubTrail: Bool = false
    var showPlaneLines: Bool = false
    var showSkeleton: Bool = true
    var onAnalysisState: ((Bool, Bool, Int) -> Void)? = nil   // (poseBusy, clubBusy, punkter)

    var body: some View {
        ZStack {
            VideoPlayer(player: player)
            if showSkeleton, let cache {
                PoseOverlay(pose: currentPose,
                            videoSize: CGSize(width: cache.width, height: cache.height),
                            fill: false)
                    .allowsHitTesting(false)
            }
            if showPlaneLines, clip.view == .dtl, let cc = clubCache, !planeLines.isEmpty {
                PlaneLinesOverlay(lines: planeLines,
                                  videoSize: CGSize(width: cc.width, height: cc.height))
                    .allowsHitTesting(false)
            }
            if showClubTrail, let cc = clubCache {
                ClubTrailOverlay(cache: cc, upTo: currentTime, topIdx: clubTopIdx,
                                 videoSize: CGSize(width: cc.width, height: cc.height))
                    .allowsHitTesting(false)
            }
            if analyzing {
                VStack { ProgressView(); Text("Analyserer pose…").font(.caption).foregroundStyle(.white) }
                    .padding().background(.black.opacity(0.6)).clipShape(RoundedRectangle(cornerRadius: 10))
            }
        }
        .onAppear {
            player.replaceCurrentItem(with: AVPlayerItem(url: clip.url))
            let obs = player.addPeriodicTimeObserver(
                forInterval: CMTime(seconds: 0.03, preferredTimescale: 600), queue: .main) { time in
                let t = CMTimeGetSeconds(time)
                self.currentTime = t
                if let cache = self.cache {
                    self.currentPose = PoseAnalyzer.pose(at: t, in: cache)
                }
            }
            self.observer = obs
            PoseAnalyzer.loadOrCompute(for: clip.url) { c in
                self.cache = c; self.analyzing = false
                self.computePlaneLines()
                self.onAnalysisState?(false, self.clubAnalyzing, self.clubCache?.frames.count ?? 0)
            }
            ClubAnalyzer.loadOrCompute(for: clip.url) { cc in
                self.clubCache = cc; self.clubAnalyzing = false
                if let cc { self.clubTopIdx = ClubAnalyzer.topIndex(in: cc, impact: clip.impact) }
                self.computePlaneLines()
                self.onAnalysisState?(self.analyzing, false, cc?.frames.count ?? 0)
            }
        }
        .onDisappear {
            if let observer { player.removeTimeObserver(observer) }
            player.pause()
        }
    }

    /// Address: bold = median af klubhoved i klippets foerste 0,4s.
    /// Guld linje bold->hofte(midt), groen linje bold->trail-skulder. Statisk fra P1.
    private func computePlaneLines() {
        guard let cc = clubCache, let pc = cache else { return }
        let early = cc.frames.filter { $0.t <= 0.4 }
        let t0 = early.isEmpty ? 0.15 : early[early.count / 2].t
        let pose = PoseAnalyzer.pose(at: t0, in: pc)
        // Forventet boldposition ved address = KLUBHOVEDETS median i de foerste 0,4s
        // (v4 detekterer det paalideligt; bolden ligger klos op ad klubhovedet ved address).
        // Fallback: pose-heuristik (under haenderne, ankelhoejde) hvis ingen tidlige klub-frames.
        var expected: CGPoint? = nil
        if early.count >= 3 {
            let xs = early.map(\.x).sorted(), ys = early.map(\.y).sorted()
            expected = CGPoint(x: xs[xs.count / 2], y: ys[ys.count / 2])
        } else if let w = pose[.rightWrist] ?? pose[.leftWrist] {
            let ankleY = (pose[.rightAnkle] ?? pose[.leftAnkle])?.y ?? min(w.y + 0.25, 0.95)
            expected = CGPoint(x: w.x, y: ankleY)
        }
        let ball: CGPoint
        if let exp = expected, !cc.ballCands.isEmpty {
            let bestCand = cc.ballCands.min { a, b in
                let da = hypot(a[0] - exp.x, a[1] - exp.y) - 0.05 * a[2]
                let db = hypot(b[0] - exp.x, b[1] - exp.y) - 0.05 * b[2]
                return da < db
            }!
            ball = CGPoint(x: bestCand[0], y: bestCand[1])
        } else if !cc.ballCands.isEmpty {
            let xs = cc.ballCands.map { $0[0] }.sorted(), ys = cc.ballCands.map { $0[1] }.sorted()
            ball = CGPoint(x: xs[xs.count / 2], y: ys[ys.count / 2])
        } else if !early.isEmpty {
            let xs = early.map(\.x).sorted(), ys = early.map(\.y).sorted()
            ball = CGPoint(x: xs[xs.count / 2], y: ys[ys.count / 2])
        } else { return }
        var lines: [PlaneLine] = []
        if let rh = pose[.rightHip], let lh = pose[.leftHip] {
            let hip = CGPoint(x: (rh.x + lh.x) / 2, y: (rh.y + lh.y) / 2)
            lines.append(PlaneLine(a: ball, b: hip, color: .spGold))
        } else if let hip = pose[.rightHip] ?? pose[.leftHip] {
            lines.append(PlaneLine(a: ball, b: hip, color: .spGold))
        }
        if let sh = pose[.rightShoulder] ?? pose[.leftShoulder] {
            lines.append(PlaneLine(a: ball, b: sh, color: .green))
        }
        planeLines = lines
    }
}

// MARK: - Enkelt-afspiller

struct ClipPlayerView: View {
    let clip: Clip
    var store: ClipStore? = nil
    @State private var player = AVPlayer()
    @State private var duration: Double = 0
    @State private var current: Double = 0
    @State private var scrubbing = false
    @State private var timeObs: Any? = nil
    @State private var comment = ""
    @State private var showSkeleton = true
    @State private var fullscreen = false
    @State private var showTrail = false
    @State private var showPlane = false
    @State private var poseBusy = true
    @State private var clubBusy = true
    @State private var clubPoints = 0

    var body: some View {
        VStack {
            SkeletonVideo(clip: clip, player: player,
                          showClubTrail: showTrail, showPlaneLines: showPlane,
                          showSkeleton: showSkeleton,
                          onAnalysisState: { p, c, n in
                              poseBusy = p; clubBusy = c; clubPoints = n
                          })
                .frame(maxHeight: 520)
            VStack(spacing: 2) {
                Slider(value: Binding(
                    get: { current },
                    set: { v in
                        current = v
                        player.seek(to: CMTime(seconds: v, preferredTimescale: 600),
                                    toleranceBefore: .zero, toleranceAfter: .zero)
                    }), in: 0...max(duration, 0.01),
                    onEditingChanged: { editing in
                        scrubbing = editing
                        if editing { player.pause() }
                    })
                    .tint(.spGold)
                HStack(spacing: 24) {
                    Button { player.pause(); player.currentItem?.step(byCount: -1) } label: {
                        Image(systemName: "backward.frame").font(.title3)
                    }
                    Button { player.seek(to: .zero); player.play() } label: {
                        Image(systemName: "gobackward").font(.title2)
                    }
                    Button { player.play() } label: { Image(systemName: "play.fill").font(.title) }
                    Button { player.pause() } label: { Image(systemName: "pause.fill").font(.title2) }
                    Button { player.pause(); player.currentItem?.step(byCount: 1) } label: {
                        Image(systemName: "forward.frame").font(.title3)
                    }
                }
                Text(String(format: "%.2f s", current)).font(.caption2.monospacedDigit()).foregroundStyle(.secondary)
            }
            .padding(.horizontal, 16).padding(.top, 4)
            HStack(spacing: 10) {
                Toggle(isOn: $showSkeleton) {
                    Label("Skelet", systemImage: "figure.stand")
                        .font(.caption.weight(.semibold))
                }
                .toggleStyle(.button).tint(.spGold)
                Toggle(isOn: $showTrail) {
                    Label(clubBusy ? "Analyserer…" : "Klubhoved-spor",
                          systemImage: clubBusy ? "hourglass" : "scribble.variable")
                        .font(.caption.weight(.semibold))
                }
                .toggleStyle(.button).tint(.spGold)
                .disabled(clubBusy || ClubAnalyzer.model == nil)
                if clip.view == .dtl {
                    Toggle(isOn: $showPlane) {
                        Label((clubBusy || poseBusy) ? "Analyserer…" : "Plane lines",
                              systemImage: (clubBusy || poseBusy) ? "hourglass" : "line.diagonal")
                            .font(.caption.weight(.semibold))
                    }
                    .toggleStyle(.button).tint(.spGold)
                    .disabled(clubBusy || poseBusy)
                }
            }
            if ClubAnalyzer.model == nil {
                Text("Klubhoved-model mangler i bygget").font(.caption2).foregroundStyle(.orange)
            } else if !clubBusy {
                Text("Klubhoved: \(clubPoints) punkter").font(.caption2).foregroundStyle(.secondary)
            }
            HStack(spacing: 8) {
                Image(systemName: "text.bubble").foregroundStyle(Color.spGold)
                TextField("Kommentar til svinget…", text: $comment, axis: .vertical)
                    .textFieldStyle(.roundedBorder).lineLimit(1...3)
                    .onSubmit { store?.setComment(clip, comment) }
            }
            .padding(.horizontal, 16).padding(.top, 6)
            Spacer()
        }
        .navigationTitle(clip.view.rawValue)
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            // Del klip (fx til OneDrive) -> analyse paa PC'en
            ToolbarItem(placement: .primaryAction) {
                Button { fullscreen = true } label: {
                    Image(systemName: "arrow.up.left.and.arrow.down.right")
                }
            }
            ToolbarItem(placement: .primaryAction) {
                ShareLink(item: clip.url) { Image(systemName: "square.and.arrow.up") }
            }
        }
        .fullScreenCover(isPresented: $fullscreen) {
            ZStack(alignment: .topTrailing) {
                Color.black.ignoresSafeArea()
                SkeletonVideo(clip: clip, player: player,
                              showClubTrail: showTrail, showPlaneLines: showPlane,
                              showSkeleton: showSkeleton)
                Button { fullscreen = false } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(.system(size: 30)).foregroundStyle(.white.opacity(0.85))
                        .padding(16)
                }
                // Samme kontroller som i afspilleren - live bindinger til samme state
                VStack {
                    Spacer()
                    HStack(spacing: 8) {
                        Toggle(isOn: $showSkeleton) {
                            Image(systemName: "figure.stand").font(.caption.weight(.semibold))
                        }
                        .toggleStyle(.button).tint(.spGold)
                        Toggle(isOn: $showTrail) {
                            Image(systemName: "scribble.variable").font(.caption.weight(.semibold))
                        }
                        .toggleStyle(.button).tint(.spGold)
                        .disabled(clubBusy || ClubAnalyzer.model == nil)
                        if clip.view == .dtl {
                            Toggle(isOn: $showPlane) {
                                Image(systemName: "line.diagonal").font(.caption.weight(.semibold))
                            }
                            .toggleStyle(.button).tint(.spGold)
                            .disabled(clubBusy || poseBusy)
                        }
                        Spacer()
                        Button { player.pause(); player.currentItem?.step(byCount: -1) } label: {
                            Image(systemName: "backward.frame").font(.title3).foregroundStyle(.white)
                        }
                        Button { player.pause(); player.currentItem?.step(byCount: 1) } label: {
                            Image(systemName: "forward.frame").font(.title3).foregroundStyle(.white)
                        }
                    }
                    .padding(.horizontal, 14).padding(.vertical, 8)
                    .background(.black.opacity(0.45)).clipShape(Capsule())
                    .padding(.horizontal, 12).padding(.bottom, 8)
                }
            }
        }
        .onAppear {
            comment = clip.comment
            timeObs = player.addPeriodicTimeObserver(
                forInterval: CMTime(seconds: 0.03, preferredTimescale: 600), queue: .main) { time in
                if !scrubbing { current = CMTimeGetSeconds(time) }
            }
        }
        .task {
            let sec = (try? await AVURLAsset(url: clip.url).load(.duration).seconds) ?? 0
            if sec.isFinite && sec > 0 { duration = sec }
        }
        .onDisappear {
            if let timeObs { player.removeTimeObserver(timeObs) }
            if comment != clip.comment { store?.setComment(clip, comment) }
        }
    }
}

// MARK: - Side-om-side synkron review

struct SyncReviewView: View {
    @ObservedObject var store: ClipStore
    @Environment(\.dismiss) private var dismiss

    @State private var dtl: Clip?
    @State private var face: Clip?
    @State private var playerA = AVPlayer()
    @State private var playerB = AVPlayer()
    @State private var scrub: Double = 0

    private var dtlClips: [Clip] { store.clips.filter { $0.view == .dtl } }
    private var faceClips: [Clip] { store.clips.filter { $0.view == .faceOn } }

    var body: some View {
        NavigationStack {
            VStack {
                HStack(spacing: 4) {
                    clipColumn(title: "Face-on", clips: faceClips, selection: $face, player: playerB)
                    clipColumn(title: "DTL", clips: dtlClips, selection: $dtl, player: playerA)
                }

                VStack(spacing: 12) {
                    HStack(spacing: 34) {
                        Button { scrub = -1.5; seekBoth(to: -1.5) } label: {
                            Image(systemName: "backward.end.fill").font(.title2)
                        }
                        Button { playSynced() } label: { Image(systemName: "play.fill").font(.largeTitle) }
                        Button { pauseBoth() } label: { Image(systemName: "pause.fill").font(.title) }
                    }
                    HStack(spacing: 8) {
                        Text("−1,5s").font(.caption2).foregroundStyle(.secondary)
                        Slider(value: $scrub, in: -1.5...1.5)
                            .onChange(of: scrub) { _, v in pauseBoth(); seekBoth(to: v) }
                        Text("+1,5s").font(.caption2).foregroundStyle(.secondary)
                    }
                    Text("Ét sæt kontroller styrer begge videoer, synkroniseret på impact.")
                        .font(.caption2).foregroundStyle(.secondary)
                }
                .disabled(dtl == nil || face == nil)
                .padding()
            }
            .navigationTitle("Sammenlign")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .cancellationAction) { Button("Luk") { dismiss() } } }
        }
    }

    @ViewBuilder
    private func clipColumn(title: String, clips: [Clip], selection: Binding<Clip?>, player: AVPlayer) -> some View {
        VStack {
            if let clip = selection.wrappedValue {
                SkeletonVideo(clip: clip, player: player).frame(height: 360).id(clip.id)
            } else {
                RoundedRectangle(cornerRadius: 8).fill(.black.opacity(0.15))
                    .frame(height: 360)
                    .overlay(Text("Vælg \(title)").foregroundStyle(.secondary))
            }
            Menu {
                ForEach(clips) { c in
                    Button { selection.wrappedValue = c } label: {
                        Label {
                            Text("Sving \(c.number) · \(ClipRow.dateString(c.date))")
                        } icon: {
                            if let ui = cachedThumb(c) {
                                Image(uiImage: ui)
                            } else {
                                Image(systemName: c.view == .dtl ? "figure.golf" : "person.fill")
                            }
                        }
                    }
                }
            } label: {
                Text(selection.wrappedValue.map { "Sving \($0.number) · \(title)" } ?? "Vælg \(title)")
                    .font(.caption)
            }
        }
    }

    /// Thumbnail fra bibliotekets cache (.thumb.jpg-sidefil); nedskaleret til menu-ikon.
    private func cachedThumb(_ clip: Clip) -> UIImage? {
        let thumb = clip.url.deletingPathExtension().appendingPathExtension("thumb.jpg")
        guard let data = try? Data(contentsOf: thumb), let ui = UIImage(data: data) else { return nil }
        return ui.preparingThumbnail(of: CGSize(width: 44, height: 60)) ?? ui
    }

    /// Søg begge klip til (impact + offset) → de holdes synkrone på impact-øjeblikket.
    private func seekBoth(to offset: Double) {
        func s(_ player: AVPlayer, _ clip: Clip?) {
            let impact = clip?.impact ?? 1.5
            player.seek(to: CMTime(seconds: max(0, impact + offset), preferredTimescale: 600))
        }
        s(playerA, dtl); s(playerB, face)
    }
    private func pauseBoth() { playerA.pause(); playerB.pause() }
    private func playSynced() {
        seekBoth(to: -1.5)
        playerA.play(); playerB.play()
    }
}

// MARK: - Pose-overlay

struct PoseOverlay: View {
    let pose: PoseDict
    let videoSize: CGSize
    var fill: Bool = true

    var body: some View {
        Canvas { ctx, size in
            guard videoSize.width > 0, videoSize.height > 0, !pose.isEmpty else { return }
            let scale = fill
                ? max(size.width / videoSize.width, size.height / videoSize.height)
                : min(size.width / videoSize.width, size.height / videoSize.height)
            let dispW = videoSize.width * scale, dispH = videoSize.height * scale
            let ox = (size.width - dispW) / 2, oy = (size.height - dispH) / 2
            func p(_ n: CGPoint) -> CGPoint { CGPoint(x: ox + n.x * dispW, y: oy + n.y * dispH) }

            for (a, b) in CameraManager.bones {
                if let pa = pose[a], let pb = pose[b] {
                    var path = Path(); path.move(to: p(pa)); path.addLine(to: p(pb))
                    ctx.stroke(path, with: .color(.green.opacity(0.9)), lineWidth: 3)
                }
            }
            for (_, n) in pose {
                let q = p(n)
                ctx.fill(Path(ellipseIn: CGRect(x: q.x - 4, y: q.y - 4, width: 8, height: 8)),
                         with: .color(.yellow))
            }
        }
    }
}

// MARK: - Klubhoved-trail (guld i tilbagesving, groen i nedsving)

struct ClubTrailOverlay: View {
    let cache: ClubCache
    let upTo: Double          // tegn kun spor frem til aktuel afspilningstid
    let topIdx: Int
    let videoSize: CGSize

    var body: some View {
        Canvas { ctx, size in
            guard videoSize.width > 0, videoSize.height > 0 else { return }
            let scale = min(size.width / videoSize.width, size.height / videoSize.height)
            let dispW = videoSize.width * scale, dispH = videoSize.height * scale
            let ox = (size.width - dispW) / 2, oy = (size.height - dispH) / 2
            func p(_ f: ClubFrame) -> CGPoint {
                CGPoint(x: ox + f.x * dispW, y: oy + f.y * dispH)
            }
            let cutoff = upTo <= 0.05 ? Double.greatestFiniteMagnitude : upTo
            let pts = cache.frames.enumerated().filter { $0.element.t <= cutoff }
            guard pts.count > 1 else { return }
            func stroke(_ seg: [CGPoint], _ color: Color) {
                guard seg.count > 1 else { return }
                var path = Path(); path.move(to: seg[0])
                for q in seg.dropFirst() { path.addLine(to: q) }
                ctx.stroke(path, with: .color(color.opacity(0.95)),
                           style: StrokeStyle(lineWidth: 4, lineCap: .round, lineJoin: .round))
            }
            let back = pts.filter { $0.offset <= topIdx }.map { p($0.element) }
            let down = pts.filter { $0.offset >= topIdx }.map { p($0.element) }
            stroke(back, .spGold)        // tilbagesving = guld
            stroke(down, .green)         // nedsving (og igennem) = groen
            if let last = pts.last {
                let q = p(last.element)
                ctx.fill(Path(ellipseIn: CGRect(x: q.x - 5, y: q.y - 5, width: 10, height: 10)),
                         with: .color(.white))
            }
        }
    }
}

// MARK: - Plane lines (statisk fra address: bold->hofte guld, bold->skulder groen)

struct PlaneLine: Identifiable {
    let id = UUID()
    let a: CGPoint            // normaliserede koordinater
    let b: CGPoint
    let color: Color
}

struct PlaneLinesOverlay: View {
    let lines: [PlaneLine]
    let videoSize: CGSize

    var body: some View {
        Canvas { ctx, size in
            guard videoSize.width > 0, videoSize.height > 0 else { return }
            let scale = min(size.width / videoSize.width, size.height / videoSize.height)
            let dispW = videoSize.width * scale, dispH = videoSize.height * scale
            let ox = (size.width - dispW) / 2, oy = (size.height - dispH) / 2
            for line in lines {
                let a = CGPoint(x: ox + line.a.x * dispW, y: oy + line.a.y * dispH)
                let b = CGPoint(x: ox + line.b.x * dispW, y: oy + line.b.y * dispH)
                // forlaeng linjen gennem hele billedet
                let dx = b.x - a.x, dy = b.y - a.y
                guard abs(dx) > 0.1 || abs(dy) > 0.1 else { continue }
                let p1 = CGPoint(x: a.x - dx * 10, y: a.y - dy * 10)
                let p2 = CGPoint(x: a.x + dx * 10, y: a.y + dy * 10)
                var path = Path(); path.move(to: p1); path.addLine(to: p2)
                // moerk kontur bagved -> synlig mod baade lys himmel og groent graes
                ctx.stroke(path, with: .color(.black.opacity(0.55)),
                           style: StrokeStyle(lineWidth: 6, dash: [10, 7]))
                ctx.stroke(path, with: .color(line.color.opacity(0.95)),
                           style: StrokeStyle(lineWidth: 3.5, dash: [10, 7]))
            }
        }
    }
}

// MARK: - Lyd-meter

struct AudioMeter: View {
    let level: Float
    let impact: Bool
    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .bottom) {
                Capsule().fill(.black.opacity(0.4))
                Capsule().fill(impact ? Color.red : Color.green)
                    .frame(height: geo.size.height * CGFloat(min(max(level, 0), 1)))
            }
        }.clipShape(Capsule())
    }
}

// MARK: - Framing-tjekliste

struct FramingChecklist: View {
    @ObservedObject var camera: CameraManager
    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            row("Telefon i vater", camera.isLevel)
            row("Hele kroppen i frame", camera.bodyInFrame)
            row(camera.distanceHint.isEmpty ? "God afstand" : camera.distanceHint, camera.distanceOK)
            row("Plads til køllen", camera.clubHeadroom)
            row("Vinkel: \(camera.detectedAngle)", camera.angleMatches)
            if camera.framingReady { Text("KLAR").font(.caption.bold()).foregroundStyle(.green) }
        }
        .padding(.horizontal, 12).padding(.vertical, 8)
        .background(.black.opacity(0.5)).clipShape(RoundedRectangle(cornerRadius: 10))
    }
    private func row(_ text: String, _ ok: Bool) -> some View {
        HStack(spacing: 6) {
            Image(systemName: ok ? "checkmark.circle.fill" : "xmark.circle").foregroundStyle(ok ? .green : .red)
            Text(text).font(.caption).foregroundStyle(.white)
        }
    }
}

// MARK: - Kamera-preview

struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession
    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.videoPreviewLayer.session = session
        view.videoPreviewLayer.videoGravity = .resizeAspectFill
        return view
    }
    func updateUIView(_ uiView: PreviewView, context: Context) {}
    final class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var videoPreviewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
    }
}

// MARK: - Kamera-motor

final class CameraManager: NSObject, ObservableObject {

    @Published var isRecording = false
    @Published var permissionGranted = false
    @Published var statusText = "Anmoder om adgang…"
    @Published var activeFPS: Double = 0
    @Published var activeResolution = ""
    @Published var isFrontCamera = true
    @Published var videoPortraitSize = CGSize(width: 1080, height: 1920)
    @Published var autoMode = true { didSet { if oldValue != autoMode { autoModeChanged() } } }
    @Published var autoListening = false

    @Published var pose: PoseDict = [:]
    @Published var poseInfo = "Pose: —"

    @Published var isLevel = false
    @Published var bodyInFrame = false
    @Published var centered = false
    @Published var distanceOK = false
    @Published var distanceHint = ""
    @Published var clubHeadroom = false
    @Published var detectedAngle = "—"
    @Published var angleMatches = false
    @Published var selectedView: SwingView = .faceOn { didSet { recomputeFraming() } }
    var framingReady: Bool { isLevel && bodyInFrame && distanceOK && clubHeadroom && angleMatches }
    var allChecksPass: Bool { framingReady && centered }
    var insideBox: Bool { bodyInFrame && distanceOK && clubHeadroom && centered }
    var frameHint: String {
        if !bodyInFrame { return "Få hele kroppen i billedet" }
        if !isLevel { return "Sæt telefonen i vater" }
        if !angleMatches { return "Vinklen ligner ikke \(selectedView.rawValue)" }
        if !centered { return "Stil dig i midten" }
        if !distanceOK { return distanceHint }
        if !clubHeadroom { return "Plads til køllen over hovedet" }
        return ""
    }

    @Published var audioLevel: Float = 0
    @Published var impactFlash = false
    @Published var windSensitivity: WindSensitivity = .normal
    @Published var poseGateEnabled = true   // lyd taeller kun som impact hvis haenderne var i fart
    var noiseFloor: Float = 0.02
    var lastImpactAt: Double = 0
    var lastFastHandsAt: Double = 0
    private var lastHandPts: [VNHumanBodyPoseObservation.JointName: (CGPoint, Double)] = [:]

    var onClipSaved: (() -> Void)?

    let session = AVCaptureSession()

    private let movieOutput = AVCaptureMovieFileOutput()
    private let videoDataOutput = AVCaptureVideoDataOutput()
    private let audioDataOutput = AVCaptureAudioDataOutput()
    private let sessionQueue = DispatchQueue(label: "swingplane.session")
    private let videoQueue = DispatchQueue(label: "swingplane.video")
    private let audioQueue = DispatchQueue(label: "swingplane.audio")
    private var videoDeviceInput: AVCaptureDeviceInput?
    private let poseRequest = VNDetectHumanBodyPoseRequest()
    private var lastPoseTime = CFAbsoluteTimeGetCurrent()
    private var lastAudioUI = CFAbsoluteTimeGetCurrent()
    private let motion = CMMotionManager()

    private var recordingView: SwingView = .faceOn
    private var recordingIsAuto = false
    private var impactSeconds: Double? = nil     // impact-tid i den aktuelle optagelse
    private var armGeneration = 0
    private var suppressRearm = false

    private var startPosition: AVCaptureDevice.Position { isFrontCamera ? .front : .back }
    private var visionOrientation: CGImagePropertyOrientation { isFrontCamera ? .leftMirrored : .right }

    static let bones: [(VNHumanBodyPoseObservation.JointName, VNHumanBodyPoseObservation.JointName)] = [
        (.neck, .nose), (.neck, .leftShoulder), (.neck, .rightShoulder),
        (.leftShoulder, .leftElbow), (.leftElbow, .leftWrist),
        (.rightShoulder, .rightElbow), (.rightElbow, .rightWrist),
        (.leftShoulder, .leftHip), (.rightShoulder, .rightHip), (.leftHip, .rightHip),
        (.leftHip, .leftKnee), (.leftKnee, .leftAnkle),
        (.rightHip, .rightKnee), (.rightKnee, .rightAnkle)
    ]

    // MARK: Start
    func start() {
        Task {
            let cam = await AVCaptureDevice.requestAccess(for: .video)
            let mic = await AVCaptureDevice.requestAccess(for: .audio)
            await MainActor.run { self.permissionGranted = cam && mic }
            if cam && mic { sessionQueue.async { self.configureSession() }; startMotion() }
            else { setStatus("Kamera-/mikrofon-adgang nægtet") }
        }
    }

    private func configureSession() {
        session.beginConfiguration()
        guard addVideoInput(position: startPosition) else {
            session.commitConfiguration(); setStatus("Kunne ikke tilgå kameraet"); return
        }
        if let audioDevice = AVCaptureDevice.default(for: .audio),
           let audioInput = try? AVCaptureDeviceInput(device: audioDevice),
           session.canAddInput(audioInput) { session.addInput(audioInput) }

        if session.canAddOutput(movieOutput) { session.addOutput(movieOutput) }
        videoDataOutput.alwaysDiscardsLateVideoFrames = true
        videoDataOutput.setSampleBufferDelegate(self, queue: videoQueue)
        if session.canAddOutput(videoDataOutput) { session.addOutput(videoDataOutput) }
        audioDataOutput.setSampleBufferDelegate(self, queue: audioQueue)
        if session.canAddOutput(audioDataOutput) { session.addOutput(audioDataOutput) }

        session.commitConfiguration()
        if let device = videoDeviceInput?.device { configureHighFrameRate(device: device) }
        session.startRunning()
        if autoMode { armInternal() }   // håndfri: start med at lytte med det samme
    }

    private func addVideoInput(position: AVCaptureDevice.Position) -> Bool {
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: position),
              let input = try? AVCaptureDeviceInput(device: device),
              session.canAddInput(input) else { return false }
        session.addInput(input); videoDeviceInput = input
        return true
    }

    private func disableMirrorOnMovie() {
        if let c = movieOutput.connection(with: .video), c.isVideoMirroringSupported {
            c.automaticallyAdjustsVideoMirroring = false
            c.isVideoMirrored = false
        }
    }

    // MARK: Flip
    func flipCamera() {
        sessionQueue.async {
            guard let current = self.videoDeviceInput else { return }
            let newPos: AVCaptureDevice.Position = (current.device.position == .front) ? .back : .front
            self.session.beginConfiguration()
            self.session.removeInput(current)
            if self.addVideoInput(position: newPos) {
                Task { @MainActor in self.isFrontCamera = (newPos == .front) }
            } else { self.session.addInput(current); self.videoDeviceInput = current }
            self.session.commitConfiguration()
            if let device = self.videoDeviceInput?.device { self.configureHighFrameRate(device: device) }
        }
    }

    // MARK: Format
    private func configureHighFrameRate(device: AVCaptureDevice, cap: Double = 240) {
        func maxRate(_ f: AVCaptureDevice.Format) -> Double {
            f.videoSupportedFrameRateRanges.map(\.maxFrameRate).max() ?? 0
        }
        func rank(_ f: AVCaptureDevice.Format) -> (Int, Double, Int32) {
            let d = CMVideoFormatDescriptionGetDimensions(f.formatDescription)
            return (d.width >= 1920 ? 1 : 0, maxRate(f), d.width)
        }
        guard let chosen = device.formats.max(by: { a, b in
            let ra = rank(a), rb = rank(b)
            if ra.0 != rb.0 { return ra.0 < rb.0 }
            if ra.1 != rb.1 { return ra.1 < rb.1 }
            return ra.2 < rb.2
        }) else { return }
        do {
            try device.lockForConfiguration()
            device.activeFormat = chosen
            let fps = min(cap, maxRate(chosen))
            let dur = CMTime(value: 1, timescale: CMTimeScale(fps))
            device.activeVideoMinFrameDuration = dur
            device.activeVideoMaxFrameDuration = dur
            // Frys koellen: loft over lukketiden (maks 1/1000s). Autoeksponering
            // koerer stadig og kompenserer med ISO — men maa aldrig vaelge lang
            // lukketid i graavejr (= motion blur = klubhoved-detektion doer).
            let maxShutter = CMTime(value: 1, timescale: 1000)
            if device.activeFormat.minExposureDuration <= maxShutter {
                device.activeMaxExposureDuration = maxShutter
            }
            device.unlockForConfiguration()
            let dims = CMVideoFormatDescriptionGetDimensions(chosen.formatDescription)
            let cam = (device.position == .front) ? "Front" : "Bag"
            let portrait = CGSize(width: CGFloat(min(dims.width, dims.height)),
                                  height: CGFloat(max(dims.width, dims.height)))
            Task { @MainActor in
                self.activeFPS = fps; self.activeResolution = "\(dims.width)×\(dims.height)"
                self.videoPortraitSize = portrait
                self.statusText = "\(cam) – \(Int(fps)) fps @ \(dims.width)×\(dims.height)"
            }
        } catch { setStatus("Kunne ikke sætte fps: \(error.localizedDescription)") }
    }

    // MARK: Vater
    private func startMotion() {
        guard motion.isDeviceMotionAvailable else { return }
        motion.deviceMotionUpdateInterval = 0.1
        motion.startDeviceMotionUpdates(to: .main) { [weak self] m, _ in
            guard let self, let m else { return }
            let roll = atan2(m.gravity.x, -m.gravity.y) * 180 / .pi
            self.isLevel = abs(roll) < 4
            self.recomputeFraming()
        }
    }

    // MARK: Optagelse (manuel + håndfri auto)

    /// Knappen: i auto = pause/genoptag lytning; i manuel = start/stop optagelse.
    func toggleRecording() {
        if autoMode {
            sessionQueue.async {
                if self.movieOutput.isRecording {
                    self.suppressRearm = true
                    self.movieOutput.stopRecording()      // pause (kasseres, re-armer ikke)
                } else {
                    self.armInternal()
                }
            }
        } else {
            let view = selectedView
            sessionQueue.async {
                if self.movieOutput.isRecording {
                    self.movieOutput.stopRecording()
                } else {
                    self.recordingView = view
                    self.recordingIsAuto = false
                    self.impactSeconds = nil
                    self.disableMirrorOnMovie()
                    let url = FileManager.default.temporaryDirectory
                        .appendingPathComponent("rec_\(Int(Date().timeIntervalSince1970)).mov")
                    self.movieOutput.startRecording(to: url, recordingDelegate: self)
                }
            }
        }
    }

    func arm() { sessionQueue.async { self.armInternal() } }

    /// Bibliotek aabnes: pause lytningen (kun i auto/Range session; manuel optagelse roeres ikke).
    func pauseListening() {
        sessionQueue.async {
            guard self.autoMode, self.movieOutput.isRecording else { return }
            self.suppressRearm = true
            self.movieOutput.stopRecording()
        }
    }
    /// Bibliotek lukkes: genoptag lytningen hvis Range session stadig er aktiv.
    func resumeListening() {
        sessionQueue.async {
            if self.autoMode && !self.movieOutput.isRecording { self.armInternal() }
        }
    }

    private func autoModeChanged() {
        sessionQueue.async {
            if self.autoMode {
                if !self.movieOutput.isRecording { self.armInternal() }
            } else if self.movieOutput.isRecording {
                self.suppressRearm = true
                self.movieOutput.stopRecording()
            }
        }
    }

    /// Arm håndfri optagelse: optag kontinuerligt, lyt efter impact, re-arm efter hvert sving.
    private func armInternal() {
        guard !movieOutput.isRecording else { return }
        armGeneration += 1
        let gen = armGeneration
        recordingIsAuto = true
        recordingView = selectedView
        impactSeconds = nil
        suppressRearm = false
        disableMirrorOnMovie()
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("rec_\(Int(Date().timeIntervalSince1970)).mov")
        movieOutput.startRecording(to: url, recordingDelegate: self)
        Task { @MainActor in self.autoListening = true; self.statusText = "LYTTER – slå dit sving" }
        // Genbrug buffer hvis intet impact inden 30s (undgå kæmpe temp-fil)
        sessionQueue.asyncAfter(deadline: .now() + 30) {
            if gen == self.armGeneration, self.movieOutput.isRecording, self.impactSeconds == nil {
                self.movieOutput.stopRecording()
            }
        }
    }

    /// Kaldes fra audio-tråden når impact høres i auto-tilstand.
    private func handleAutoImpact() {
        sessionQueue.async {
            guard self.recordingIsAuto, self.movieOutput.isRecording, self.impactSeconds == nil else { return }
            self.impactSeconds = CMTimeGetSeconds(self.movieOutput.recordedDuration)
            Task { @MainActor in self.autoListening = false; self.statusText = "Sving fanget – gemmer…" }
            self.sessionQueue.asyncAfter(deadline: .now() + 1.2) {
                if self.movieOutput.isRecording { self.movieOutput.stopRecording() }
            }
        }
    }

    // MARK: Framing
    func recomputeFraming() {
        guard !pose.isEmpty else {
            bodyInFrame = false; distanceOK = false; distanceHint = ""; clubHeadroom = false; centered = false; angleMatches = false; return
        }
        let xs = pose.values.map { $0.x }, ys = pose.values.map { $0.y }
        guard let minX = xs.min(), let maxX = xs.max(), let minY = ys.min(), let maxY = ys.max() else { return }
        let hasHead = pose[.nose] != nil || pose[.neck] != nil
        let hasFeet = pose[.leftAnkle] != nil || pose[.rightAnkle] != nil
        bodyInFrame = hasHead && hasFeet && minX > 0.03 && maxX < 0.97 && minY > 0.02 && maxY < 0.98
        let bcx = (minX + maxX) / 2
        centered = bcx > 0.34 && bcx < 0.66
        let height = maxY - minY
        // Front-kamera: bredere baand (man staar laengere vaek for at kunne se skaermen)
        let minH: CGFloat = isFrontCamera ? 0.26 : 0.42
        let maxH: CGFloat = isFrontCamera ? 0.82 : 0.72
        if height < minH { distanceOK = false; distanceHint = "Træd tættere på" }
        else if height > maxH { distanceOK = false; distanceHint = "Træd tilbage" }
        else { distanceOK = true; distanceHint = "" }
        clubHeadroom = minY > 0.12
        if let ls = pose[.leftShoulder], let rs = pose[.rightShoulder], height > 0.01 {
            let ratio = abs(ls.x - rs.x) / height
            if ratio > 0.28 { detectedAngle = "Face-on" }
            else if ratio < 0.14 { detectedAngle = "DTL" }
            else { detectedAngle = "Diagonal" }
        } else { detectedAngle = "—" }
        angleMatches = (detectedAngle == selectedView.rawValue)
    }

    private func setStatus(_ text: String) { Task { @MainActor in self.statusText = text } }
}

// MARK: - Sample-buffer-delegate

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate,
                         AVCaptureAudioDataOutputSampleBufferDelegate {

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        if output === videoDataOutput { processVideo(sampleBuffer) }
        else if output === audioDataOutput { processAudio(sampleBuffer) }
    }

    private func processVideo(_ sampleBuffer: CMSampleBuffer) {
        let now = CFAbsoluteTimeGetCurrent()
        guard now - lastPoseTime > 0.04 else { return }
        lastPoseTime = now
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: visionOrientation, options: [:])
        do {
            try handler.perform([poseRequest])
            guard let obs = poseRequest.results?.first else {
                Task { @MainActor in self.pose = [:]; self.poseInfo = "Pose: ingen person"; self.recomputeFraming() }
                return
            }
            let pts = (try? obs.recognizedPoints(.all)) ?? [:]
            var result: PoseDict = [:]; var confSum: Float = 0; var count = 0
            for (name, pt) in pts where pt.confidence > 0.3 {
                result[name] = CGPoint(x: pt.location.x, y: 1 - pt.location.y); confSum += pt.confidence; count += 1
            }
            let avg = count > 0 ? confSum / Float(count) : 0
            // Pose-gate: registrer hurtig haand/arm-bevaegelse (sving i gang).
            // Wrist OG elbow (elbow er mere robust ved motion blur i nedsvinget).
            for j in [VNHumanBodyPoseObservation.JointName.rightWrist, .leftWrist, .rightElbow, .leftElbow] {
                if let p = result[j] {
                    if let (prev, t) = lastHandPts[j], now - t < 0.25, now > t {
                        let speed = hypot(p.x - prev.x, p.y - prev.y) / CGFloat(now - t)
                        if speed > 1.2 { lastFastHandsAt = now }   // ~>1.2 skaermbredde/s = sving
                    }
                    lastHandPts[j] = (p, now)
                }
            }
            Task { @MainActor in
                self.pose = result
                self.poseInfo = count > 0 ? "Pose: \(count)/19 led · konf \(String(format: "%.2f", avg))" : "Pose: ingen person"
                self.recomputeFraming()
            }
        } catch { }
    }

    private func processAudio(_ sampleBuffer: CMSampleBuffer) {
        let peak = audioPeak(sampleBuffer)
        let isImpact = detectImpact(peak)
        if isImpact { handleAutoImpact() }
        let now = CFAbsoluteTimeGetCurrent()
        guard now - lastAudioUI > 0.05 else { return }
        lastAudioUI = now
        Task { @MainActor in
            self.audioLevel = peak
            if isImpact {
                self.impactFlash = true
                Task { try? await Task.sleep(nanoseconds: 150_000_000); await MainActor.run { self.impactFlash = false } }
            }
        }
    }

    // Vind-fix (mulighed 2): adaptiv transient-detektion. En spids skal springe
    // markant over et GLIDENDE baggrundsniveau OG over et absolut gulv. Vindstød er
    // vedvarende → de hæver baggrunden og rammer derfor ikke ratio-kravet; et golfslag
    // er en kort, kraftig transient → passerer. Debounce 0,6s mod dobbelt-trig.
    private func detectImpact(_ peak: Float) -> Bool {
        let now = CFAbsoluteTimeGetCurrent()
        let ratio = windSensitivity.ratio
        let floor = windSensitivity.floor
        let soundOK = peak > floor && peak > noiseFloor * ratio && (now - lastImpactAt) > 0.6
        let handsOK = !poseGateEnabled || (now - lastFastHandsAt) < 0.5
        let isImpact = soundOK && handsOK
        if isImpact { lastImpactAt = now }
        else { noiseFloor = noiseFloor * 0.97 + peak * 0.03 }
        return isImpact
    }

    private func audioPeak(_ sampleBuffer: CMSampleBuffer) -> Float {
        guard let fd = CMSampleBufferGetFormatDescription(sampleBuffer),
              let asbdPtr = CMAudioFormatDescriptionGetStreamBasicDescription(fd) else { return 0 }
        let asbd = asbdPtr.pointee
        let isFloat = (asbd.mFormatFlags & kAudioFormatFlagIsFloat) != 0
        let bytesPerChannel = Int(asbd.mBitsPerChannel / 8)
        var blockBuffer: CMBlockBuffer?
        var abl = AudioBufferList()
        let status = CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(
            sampleBuffer, bufferListSizeNeededOut: nil, bufferListOut: &abl,
            bufferListSize: MemoryLayout<AudioBufferList>.size,
            blockBufferAllocator: nil, blockBufferMemoryAllocator: nil,
            flags: 0, blockBufferOut: &blockBuffer)
        guard status == noErr else { return 0 }
        var peak: Float = 0
        for buffer in UnsafeMutableAudioBufferListPointer(&abl) {
            guard let mData = buffer.mData else { continue }
            let byteCount = Int(buffer.mDataByteSize)
            if isFloat && bytesPerChannel == 4 {
                let ptr = mData.assumingMemoryBound(to: Float32.self)
                for i in 0..<(byteCount / 4) { let v = abs(ptr[i]); if v > peak { peak = v } }
            } else if bytesPerChannel == 2 {
                let ptr = mData.assumingMemoryBound(to: Int16.self)
                for i in 0..<(byteCount / 2) { let v = abs(Float(ptr[i]) / 32768); if v > peak { peak = v } }
            }
        }
        return min(peak, 1)
    }
}

// MARK: - Optage-delegate (gem / auto-trim)

extension CameraManager: AVCaptureFileOutputRecordingDelegate {
    func fileOutput(_ output: AVCaptureFileOutput, didStartRecordingTo fileURL: URL,
                    from connections: [AVCaptureConnection]) {
        Task { @MainActor in self.isRecording = true; if !self.recordingIsAuto { self.statusText = "Optager…" } }
    }

    func fileOutput(_ output: AVCaptureFileOutput, didFinishRecordingTo outputFileURL: URL,
                    from connections: [AVCaptureConnection], error: Error?) {
        Task { @MainActor in self.isRecording = false }
        let view = recordingView
        if recordingIsAuto, let impact = impactSeconds {
            trimAndSave(src: outputFileURL, view: view, impact: impact)   // re-armer i completion
        } else if recordingIsAuto {
            // Auto uden impact (timeout eller bruger satte på pause) → kassér
            try? FileManager.default.removeItem(at: outputFileURL)
            let shouldRearm = autoMode && !suppressRearm
            suppressRearm = false
            if shouldRearm { arm() }
            else { Task { @MainActor in self.autoListening = false; self.statusText = "Auto på pause – tryk for at lytte" } }
        } else {
            let dest = ClipStore.newURL(view: view, impactMs: nil)
            try? FileManager.default.moveItem(at: outputFileURL, to: dest)
            Task { @MainActor in self.statusText = "Gemt ✓ (\(view.rawValue))"; self.onClipSaved?() }
        }
    }

    /// Trim til [impact-2s, impact+1.2s] uden re-encode (passthrough) → bevar fps.
    private func trimAndSave(src: URL, view: SwingView, impact: Double) {
        let asset = AVURLAsset(url: src)
        let total = CMTimeGetSeconds(asset.duration)
        let start = max(0, impact - 2.2)      // lidt mere optakt (0,2s tidligere)
        let end = min(total, impact + 1.2)
        let impactMs = Int((impact - start) * 1000)
        let dest = ClipStore.newURL(view: view, impactMs: impactMs)

        guard let export = AVAssetExportSession(asset: asset, presetName: AVAssetExportPresetPassthrough) else {
            try? FileManager.default.moveItem(at: src, to: dest)
            Task { @MainActor in self.statusText = "Gemt ✓"; self.onClipSaved?() }
            if autoMode { arm() }
            return
        }
        export.outputURL = dest
        export.outputFileType = .mov
        export.timeRange = CMTimeRange(start: CMTime(seconds: start, preferredTimescale: 600),
                                       end: CMTime(seconds: end, preferredTimescale: 600))
        export.exportAsynchronously {
            try? FileManager.default.removeItem(at: src)
            Task { @MainActor in
                if export.status == .completed {
                    self.statusText = "Sving gemt ✓ (\(view.rawValue), auto-trim)"
                } else {
                    self.statusText = "Trim-fejl: \(export.error?.localizedDescription ?? "ukendt")"
                }
                self.onClipSaved?()
            }
            if self.autoMode { self.arm() }     // re-arm til næste sving
        }
    }
}

// MARK: - Design-batch (2026-07-06): session-mode + åbningsflow + vinkelvalg
// UI-only lag foran kameraet – kan køres i iOS Simulator UDEN kamera.

enum SessionMode: String { case single = "Ét sving", multiple = "Træningssession" }

enum FocusArea: String, CaseIterable, Identifiable {
    case takeaway = "Takeaway", top = "Top", impact = "Impact", follow = "Follow-through"
    var id: String { rawValue }
}
enum TrainingMode: String, CaseIterable, Identifiable {
    case driver = "Driver", iron = "Jern", wedge = "Wedge", putt = "Putt"
    var id: String { rawValue }
}

private enum FlowStep { case start, angle, camera }

// Farver fra Plane-mockuppet ("Plane mockup.html": moerkegroen + guld) + diagram-farver
private extension Color {
    static let spBgTop   = Color(red: 0.110, green: 0.200, blue: 0.145) // #1c3325
    static let spBg      = Color(red: 0.039, green: 0.110, blue: 0.071) // #0a1c12
    static let spBgDeep  = Color(red: 0.020, green: 0.051, blue: 0.031) // #050d08
    static let spSurface = Color(red: 0.078, green: 0.208, blue: 0.165) // #14352a
    static let spGold    = Color(red: 0.831, green: 0.686, blue: 0.216) // #d4af37
    static let spGoldLight = Color(red: 0.941, green: 0.780, blue: 0.369) // #f0c75e
    static let spInk     = Color(red: 0.102, green: 0.102, blue: 0.039) // #1a1a0a
    static let spCream   = Color(red: 0.961, green: 0.945, blue: 0.902) // #f5f1e6
    static let spPurple = Color(red: 0.50, green: 0.47, blue: 0.87) // #7f77dd
    static let spAmber  = Color(red: 0.73, green: 0.46, blue: 0.09) // #ba7517
    static let spPurpleDark = Color(red: 0.235, green: 0.204, blue: 0.537) // #3c3489
    static let spAmberLight = Color(red: 0.937, green: 0.624, blue: 0.153) // #ef9f27
}

private struct FlowBackground: View {
    var body: some View {
        RadialGradient(colors: [Color.spBgTop, Color.spBg, Color.spBgDeep],
                       center: .top, startRadius: 0, endRadius: 900)
            .ignoresSafeArea()
    }
}

struct ContentView: View {
    @State private var step: FlowStep = .start
    @State private var mode: SessionMode = .multiple
    @State private var angle: SwingView = .dtl
    @StateObject private var rootStore = ClipStore()
    @State private var showLibrary = false

    var body: some View {
        ZStack {
            switch step {
            case .start:
                OnboardingView(onSelect: { m in mode = m; step = .angle },
                               onLibrary: { showLibrary = true })
                    .transition(.opacity)
            case .angle:
                AngleSelectView(onPick: { a in angle = a; step = .camera },
                                onBack: { step = .start })
                    .transition(.opacity)
            case .camera:
                CameraScreen(mode: mode, angle: angle, onExit: { step = .start })
                    .transition(.opacity)
            }
        }
        .animation(.easeInOut(duration: 0.25), value: step)
        .sheet(isPresented: $showLibrary) { LibraryView(store: rootStore) }
    }
}

struct OnboardingView: View {
    var onSelect: (SessionMode) -> Void
    var onLibrary: () -> Void = {}
    @State private var showTutorial = false
    var body: some View {
        ZStack {
            FlowBackground()
            VStack(spacing: 18) {
                Spacer()
                VStack(spacing: 4) {
                    Text("SwingPlane").font(.largeTitle.weight(.semibold)).foregroundStyle(.white)
                    Text("Hvad vil du lave?").font(.subheadline).foregroundStyle(.white.opacity(0.6))
                }
                .padding(.bottom, 10)
                choiceCard(icon: "viewfinder", title: "Ét sving",
                           sub: "Kig på ét slag i detaljer") { onSelect(.single) }
                choiceCard(icon: "repeat", title: "Træningssession",
                           sub: "Slå en spand – hvert slag fanges automatisk") { onSelect(.multiple) }
                Button { showTutorial = true } label: {
                    Label("Sådan virker det", systemImage: "questionmark.circle")
                        .font(.subheadline.weight(.medium))
                        .foregroundStyle(Color.spGold.opacity(0.9))
                }
                .buttonStyle(.plain)
                .padding(.top, 6)
                Button { onLibrary() } label: {
                    Label("My swings", systemImage: "square.stack.3d.up.fill")
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(.white)
                        .padding(.horizontal, 18).padding(.vertical, 10)
                        .background(Color.spSurface)
                        .clipShape(Capsule())
                        .overlay(Capsule().stroke(Color.spGold.opacity(0.4), lineWidth: 1))
                }
                .buttonStyle(.plain)
                .padding(.top, 2)
                Spacer()
                Text("Version \(Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "?") · Build \(Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "?")")
                    .font(.caption2).foregroundStyle(.white.opacity(0.35))
                    .padding(.bottom, 4)
            }
            .padding(.horizontal, 24)
            .sheet(isPresented: $showTutorial) { TutorialView() }
        }
    }
    private func choiceCard(icon: String, title: String, sub: String,
                            action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 14) {
                Image(systemName: icon).font(.system(size: 26, weight: .semibold))
                    .foregroundStyle(Color.spGold).frame(width: 40)
                VStack(alignment: .leading, spacing: 3) {
                    Text(title).font(.headline).foregroundStyle(.white)
                    Text(sub).font(.subheadline).foregroundStyle(.white.opacity(0.6))
                        .fixedSize(horizontal: false, vertical: true)
                }
                Spacer()
                Image(systemName: "chevron.right").foregroundStyle(.white.opacity(0.3))
            }
            .padding(16)
            .background(Color.spSurface)
            .clipShape(RoundedRectangle(cornerRadius: 16))
            .overlay(RoundedRectangle(cornerRadius: 16).stroke(Color.spGold.opacity(0.25), lineWidth: 1))
        }
        .buttonStyle(.plain)
    }
}

struct TutorialView: View {
    @Environment(\.dismiss) private var dismiss
    var body: some View {
        ZStack {
            FlowBackground()
            VStack(alignment: .leading, spacing: 22) {
                HStack {
                    Text("Sådan virker det").font(.title2.weight(.semibold)).foregroundStyle(.white)
                    Spacer()
                    Button { dismiss() } label: {
                        Image(systemName: "xmark.circle.fill").font(.title2)
                            .foregroundStyle(.white.opacity(0.5))
                    }
                }
                .padding(.bottom, 4)
                step(num: 1, icon: "hand.tap", title: "Vælg hvad du vil lave",
                     text: "Ét sving til detaljer – eller Træningssession, hvor hvert slag fanges automatisk.")
                step(num: 2, icon: "iphone.gen3", title: "Stil telefonen",
                     text: "Vælg vinkel: bagfra (Down the Line) eller forfra (Face-on). Brug gerne et stativ.")
                step(num: 3, icon: "person.crop.rectangle", title: "Stil dig i boksen",
                     text: "Flyt dig til rammen bliver grøn – så er afstand og placering rigtig.")
                step(num: 4, icon: "figure.golf", title: "Bare sving",
                     text: "Appen hører slaget og gemmer kun selve svinget. Se dine klip under \"My swings\".")
                Spacer()
            }
            .padding(24)
        }
        .presentationDetents([.medium, .large])
    }
    private func step(num: Int, icon: String, title: String, text: String) -> some View {
        HStack(alignment: .top, spacing: 14) {
            ZStack {
                Circle().fill(Color.spSurface).frame(width: 40, height: 40)
                    .overlay(Circle().stroke(Color.spGold.opacity(0.4), lineWidth: 1))
                Image(systemName: icon).font(.system(size: 17, weight: .semibold))
                    .foregroundStyle(Color.spGold)
            }
            VStack(alignment: .leading, spacing: 3) {
                Text("\(num). \(title)").font(.headline).foregroundStyle(.white)
                Text(text).font(.subheadline).foregroundStyle(.white.opacity(0.65))
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }
}

struct AngleSelectView: View {
    var onPick: (SwingView) -> Void
    var onBack: () -> Void
    var body: some View {
        ZStack {
            FlowBackground()
            VStack(spacing: 18) {
                HStack {
                    Button(action: onBack) {
                        Image(systemName: "chevron.left").font(.title3.weight(.semibold))
                            .foregroundStyle(.white.opacity(0.8))
                    }
                    Spacer()
                }
                Spacer()
                Text("Hvor står telefonen?").font(.title2.weight(.semibold)).foregroundStyle(.white)
                angleCard(angle: .dtl, title: "Bagfra · Down the Line", sub: "På slaglinjen, bag dig")
                angleCard(angle: .faceOn, title: "Forfra · Face-on",
                          sub: "Vinkelret, foran dig – lidt længere væk")
                Spacer()
            }
            .padding(.horizontal, 24)
        }
    }
    private func angleCard(angle: SwingView, title: String, sub: String) -> some View {
        Button { onPick(angle) } label: {
            VStack(alignment: .leading, spacing: 10) {
                AngleDiagram(angle: angle).frame(height: 92)
                Text(title).font(.headline).foregroundStyle(.white)
                Text(sub).font(.subheadline).foregroundStyle(.white.opacity(0.6))
            }
            .padding(16)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color.spSurface)
            .clipShape(RoundedRectangle(cornerRadius: 16))
            .overlay(RoundedRectangle(cornerRadius: 16).stroke(Color.spGold.opacity(0.25), lineWidth: 1))
        }
        .buttonStyle(.plain)
    }
}

// Top-down skematik. Skuldre (lilla) ligger OVER slaglinjen (Alex' rettelse 2026-07-06).

// MARK: - Vind-følsomhed (adaptiv transient-tærskel)

enum WindSensitivity: String, CaseIterable, Identifiable {
    case sensitive = "Følsom", normal = "Normal", windy = "Blæst"
    var id: String { rawValue }
    var ratio: Float { self == .sensitive ? 3.5 : (self == .normal ? 5.0 : 7.0) }
    var floor: Float { self == .sensitive ? 0.08 : (self == .normal ? 0.12 : 0.18) }
}

// MARK: - Framing-boks (grøn når spilleren står inde/centreret, ellers rød)

struct FramingBox: View {
    let inside: Bool
    let hint: String
    var body: some View {
        GeometryReader { geo in
            let w = geo.size.width, h = geo.size.height
            let bw = w * 0.48, bh = h * 0.80
            let cx = w * 0.5, cy = h * 0.5
            ZStack {
                RoundedRectangle(cornerRadius: 18)
                    .stroke(inside ? Color.green : Color.red,
                            style: StrokeStyle(lineWidth: 5, lineJoin: .round, dash: inside ? [] : [12, 8]))
                    .shadow(color: (inside ? Color.green : Color.red).opacity(0.8), radius: 10)
                    .frame(width: bw, height: bh)
                    .position(x: cx, y: cy)
                if !inside && !hint.isEmpty {
                    Text(hint)
                        .font(.subheadline.weight(.semibold)).foregroundStyle(.white)
                        .padding(.horizontal, 14).padding(.vertical, 8)
                        .background(.red.opacity(0.78)).clipShape(Capsule())
                        .position(x: cx, y: cy + bh / 2 + 24)
                }
            }
        }
        .ignoresSafeArea()
    }
}

// MARK: - Vinkel-diagram (top-down, spiller m. kasket – ingen tekst)

struct AngleDiagram: View {
    let angle: SwingView
    var body: some View {
        Canvas { ctx, size in
            let k = min(size.width / 240, size.height / 130)
            let offx = (size.width - 240 * k) / 2, offy = (size.height - 130 * k) / 2
            func pt(_ x: CGFloat, _ y: CGFloat) -> CGPoint { CGPoint(x: offx + x * k, y: offy + y * k) }
            func box(_ x: CGFloat, _ y: CGFloat, _ w: CGFloat, _ h: CGFloat) -> CGRect {
                CGRect(x: offx + x * k, y: offy + y * k, width: w * k, height: h * k)
            }
            let muted = Color.white.opacity(0.32)
            let isDTL = (angle == .dtl)
            let lineY: CGFloat = isDTL ? 88 : 60
            let ballX: CGFloat = isDTL ? 150 : 120
            let px: CGFloat = isDTL ? 150 : 120
            let py: CGFloat = isDTL ? 52 : 26

            var line = Path(); line.move(to: pt(20, lineY)); line.addLine(to: pt(206, lineY))
            ctx.stroke(line, with: .color(muted), style: StrokeStyle(lineWidth: 1.5, dash: [5, 4]))
            var arrow = Path()
            arrow.move(to: pt(212, lineY)); arrow.addLine(to: pt(203, lineY - 5)); arrow.addLine(to: pt(203, lineY + 5)); arrow.closeSubpath()
            ctx.fill(arrow, with: .color(muted))

            ctx.fill(Path(ellipseIn: box(ballX - 4, lineY - 4, 8, 8)), with: .color(.white))

            ctx.fill(Path(roundedRect: box(px - 21, py - 1, 42, 8), cornerRadius: 4 * k), with: .color(.spPurple))
            ctx.fill(Path(ellipseIn: box(px - 9, py - 6, 18, 18)), with: .color(.spPurple))
            ctx.fill(Path(ellipseIn: box(px - 12, py + 4, 24, 10)), with: .color(.spPurpleDark))
            ctx.fill(Path(ellipseIn: box(px - 10, py - 12, 20, 20)), with: .color(.spPurpleDark))

            if isDTL {
                ctx.fill(Path(roundedRect: box(28, lineY - 10, 16, 20), cornerRadius: 3 * k), with: .color(.spAmber))
                var l1 = Path(); l1.move(to: pt(44, lineY - 7)); l1.addLine(to: pt(61, lineY - 16))
                var l2 = Path(); l2.move(to: pt(44, lineY + 7)); l2.addLine(to: pt(61, lineY + 16))
                ctx.stroke(l1, with: .color(.spAmberLight), style: StrokeStyle(lineWidth: 1.4 * k, lineCap: .round))
                ctx.stroke(l2, with: .color(.spAmberLight), style: StrokeStyle(lineWidth: 1.4 * k, lineCap: .round))
            } else {
                ctx.fill(Path(roundedRect: box(109, 104, 22, 14), cornerRadius: 3 * k), with: .color(.spAmber))
                var l1 = Path(); l1.move(to: pt(112, 104)); l1.addLine(to: pt(104, 88))
                var l2 = Path(); l2.move(to: pt(128, 104)); l2.addLine(to: pt(136, 88))
                ctx.stroke(l1, with: .color(.spAmberLight), style: StrokeStyle(lineWidth: 1.4 * k, lineCap: .round))
                ctx.stroke(l2, with: .color(.spAmberLight), style: StrokeStyle(lineWidth: 1.4 * k, lineCap: .round))
            }
        }
    }
}

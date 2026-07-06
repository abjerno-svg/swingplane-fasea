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

struct Clip: Identifiable {
    let id = UUID()
    let url: URL
    let view: SwingView
    let date: Date
    let impact: Double?     // sekunder inde i klippet, hvis kendt
}

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

    func reload() {
        let files = (try? FileManager.default.contentsOfDirectory(
            at: Self.directory, includingPropertiesForKeys: nil)) ?? []
        var result: [Clip] = []
        for url in files where url.pathExtension == "mov" {
            let parts = url.deletingPathExtension().lastPathComponent.split(separator: "_")
            guard parts.count >= 3, let ts = Double(parts[1]) else { continue }
            let view = SwingView.fromFileKey(String(parts[2]))
            var impact: Double? = nil
            if parts.count >= 4, let ms = Double(parts[3]) { impact = ms / 1000 }
            result.append(Clip(url: url, view: view, date: Date(timeIntervalSince1970: ts), impact: impact))
        }
        let sorted = result.sorted { $0.date > $1.date }
        // Auto-loft: slet ældste ud over maxClips
        if sorted.count > Self.maxClips {
            for c in sorted[Self.maxClips...] { removeFiles(c) }
            clips = Array(sorted.prefix(Self.maxClips))
        } else {
            clips = sorted
        }
    }

    private func removeFiles(_ clip: Clip) {
        try? FileManager.default.removeItem(at: clip.url)
        try? FileManager.default.removeItem(at: PoseAnalyzer.cacheURL(for: clip.url))
        try? FileManager.default.removeItem(at: clip.url.deletingPathExtension().appendingPathExtension("thumb.jpg"))
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

            var frames: [PoseFrame] = []
            var w = 1080.0, h = 1920.0
            let step = 1.0 / 30.0
            var t = 0.0
            while t < max(dur, 0.01) {
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

            FramingBox(inside: camera.insideBox, hint: camera.frameHint)
                .allowsHitTesting(false)

            HStack {
                AudioMeter(level: camera.audioLevel, impact: camera.impactFlash)
                    .frame(width: 8).padding(.leading, 6)
                Spacer()
            }.ignoresSafeArea(edges: .vertical)

            VStack(spacing: 8) {
                HStack {
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
                    Button { showLibrary = true } label: {
                        Image(systemName: "square.stack.3d.up.fill")
                            .font(.system(size: 20, weight: .semibold)).foregroundStyle(.white)
                            .padding(12).background(.black.opacity(0.55)).clipShape(Circle())
                    }
                    Spacer()
                    Toggle(isOn: $camera.autoMode) {
                        Label("Auto", systemImage: "bolt.fill").font(.caption.weight(.bold))
                    }
                    .toggleStyle(.button).tint(.green)
                }
                if camera.autoListening { ListeningBadge() }
                badge(camera.statusText)
                badge(camera.poseInfo)
                FramingChecklist(camera: camera).padding(.top, 4)

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
        .sheet(isPresented: $showLibrary) { LibraryView(store: store) }
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
                        .background(selected == v ? Color.green.opacity(0.85) : Color.black.opacity(0.5))
                        .foregroundStyle(.white).clipShape(Capsule())
                        .overlay(Capsule().stroke(selected == v ? Color.white : .clear, lineWidth: 1.5))
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

    private func swingNumber(_ clip: Clip) -> Int {
        let ordered = store.clips.sorted { $0.date < $1.date }
        return (ordered.firstIndex { $0.id == clip.id } ?? 0) + 1
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
                            NavigationLink { ClipPlayerView(clip: clip) } label: { ClipRow(clip: clip, number: swingNumber(clip)) }
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
    var number: Int = 0
    private static let fmt: DateFormatter = {
        let f = DateFormatter(); f.dateFormat = "d. MMM HH:mm"; return f
    }()
    static func dateString(_ d: Date) -> String { fmt.string(from: d) }
    var body: some View {
        HStack(spacing: 12) {
            ClipThumbnail(url: clip.url)
            VStack(alignment: .leading, spacing: 3) {
                Text("Sving \(number)").font(.body.weight(.semibold))
                HStack(spacing: 6) {
                    Image(systemName: clip.view == .dtl ? "figure.golf" : "person.fill")
                        .font(.caption2).foregroundStyle(.green)
                    Text(clip.view.rawValue).font(.caption).foregroundStyle(.secondary)
                }
                Text(Self.fmt.string(from: clip.date)).font(.caption2).foregroundStyle(.secondary)
            }
            Spacer()
            Image(systemName: "play.circle").foregroundStyle(.secondary)
        }
        .padding(.vertical, 2)
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

    var body: some View {
        ZStack {
            VideoPlayer(player: player)
            if let cache {
                PoseOverlay(pose: currentPose,
                            videoSize: CGSize(width: cache.width, height: cache.height),
                            fill: false)
                    .allowsHitTesting(false)
            }
            if analyzing {
                VStack { ProgressView(); Text("Analyserer pose…").font(.caption).foregroundStyle(.white) }
                    .padding().background(.black.opacity(0.6)).clipShape(RoundedRectangle(cornerRadius: 10))
            }
        }
        .onAppear {
            player.replaceCurrentItem(with: AVPlayerItem(url: clip.url))
            PoseAnalyzer.loadOrCompute(for: clip.url) { c in
                self.cache = c; self.analyzing = false
                let obs = player.addPeriodicTimeObserver(
                    forInterval: CMTime(seconds: 0.03, preferredTimescale: 600), queue: .main) { time in
                    if let cache = self.cache {
                        self.currentPose = PoseAnalyzer.pose(at: CMTimeGetSeconds(time), in: cache)
                    }
                }
                self.observer = obs
            }
        }
        .onDisappear {
            if let observer { player.removeTimeObserver(observer) }
            player.pause()
        }
    }
}

// MARK: - Enkelt-afspiller

struct ClipPlayerView: View {
    let clip: Clip
    @State private var player = AVPlayer()

    var body: some View {
        VStack {
            SkeletonVideo(clip: clip, player: player).frame(maxHeight: 520)
            HStack(spacing: 24) {
                Button { player.seek(to: .zero); player.play() } label: {
                    Image(systemName: "gobackward").font(.title2)
                }
                Button { player.play() } label: { Image(systemName: "play.fill").font(.title) }
                Button { player.pause() } label: { Image(systemName: "pause.fill").font(.title2) }
            }.padding()
            Spacer()
        }
        .navigationTitle(clip.view.rawValue)
        .navigationBarTitleDisplayMode(.inline)
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
                    Button(ClipRow.dateString(c.date)) { selection.wrappedValue = c }
                }
            } label: {
                Text(selection.wrappedValue == nil ? "Vælg \(title)" : title).font(.caption)
            }
        }
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
    var insideBox: Bool { bodyInFrame && distanceOK && clubHeadroom && centered }
    var frameHint: String {
        if !bodyInFrame { return "Få hele kroppen i billedet" }
        if !centered { return "Stil dig i midten" }
        if !distanceOK { return distanceHint }
        if !clubHeadroom { return "Plads til køllen over hovedet" }
        return ""
    }

    @Published var audioLevel: Float = 0
    @Published var impactFlash = false
    @Published var windSensitivity: WindSensitivity = .normal

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
        if height < 0.42 { distanceOK = false; distanceHint = "Træd tættere på" }
        else if height > 0.72 { distanceOK = false; distanceHint = "Træd tilbage" }
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
    private var noiseFloor: Float = 0.02
    private var lastImpactAt: Double = 0
    private func detectImpact(_ peak: Float) -> Bool {
        let now = CFAbsoluteTimeGetCurrent()
        let ratio = windSensitivity.ratio
        let floor = windSensitivity.floor
        let isImpact = peak > floor && peak > noiseFloor * ratio && (now - lastImpactAt) > 0.6
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

// Farver fra design-mockuppet
private extension Color {
    static let spTeal   = Color(red: 0.62, green: 0.88, blue: 0.80) // #9fe1cb
    static let spPurple = Color(red: 0.50, green: 0.47, blue: 0.87) // #7f77dd
    static let spAmber  = Color(red: 0.73, green: 0.46, blue: 0.09) // #ba7517
    static let spPurpleDark = Color(red: 0.235, green: 0.204, blue: 0.537) // #3c3489
    static let spAmberLight = Color(red: 0.937, green: 0.624, blue: 0.153) // #ef9f27
}

private struct FlowBackground: View {
    var body: some View {
        LinearGradient(colors: [Color(red: 0.09, green: 0.09, blue: 0.10),
                                Color(red: 0.05, green: 0.05, blue: 0.06)],
                       startPoint: .top, endPoint: .bottom)
            .ignoresSafeArea()
    }
}

struct ContentView: View {
    @State private var step: FlowStep = .start
    @State private var mode: SessionMode = .multiple
    @State private var angle: SwingView = .dtl

    var body: some View {
        ZStack {
            switch step {
            case .start:
                OnboardingView { m in mode = m; step = .angle }
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
    }
}

struct OnboardingView: View {
    var onSelect: (SessionMode) -> Void
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
                Spacer()
            }
            .padding(.horizontal, 24)
        }
    }
    private func choiceCard(icon: String, title: String, sub: String,
                            action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 14) {
                Image(systemName: icon).font(.system(size: 26, weight: .semibold))
                    .foregroundStyle(Color.spTeal).frame(width: 40)
                VStack(alignment: .leading, spacing: 3) {
                    Text(title).font(.headline).foregroundStyle(.white)
                    Text(sub).font(.subheadline).foregroundStyle(.white.opacity(0.6))
                        .fixedSize(horizontal: false, vertical: true)
                }
                Spacer()
                Image(systemName: "chevron.right").foregroundStyle(.white.opacity(0.3))
            }
            .padding(16)
            .background(Color.white.opacity(0.08))
            .clipShape(RoundedRectangle(cornerRadius: 16))
            .overlay(RoundedRectangle(cornerRadius: 16).stroke(Color.white.opacity(0.12), lineWidth: 0.5))
        }
        .buttonStyle(.plain)
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
            .background(Color.white.opacity(0.08))
            .clipShape(RoundedRectangle(cornerRadius: 16))
            .overlay(RoundedRectangle(cornerRadius: 16).stroke(Color.white.opacity(0.12), lineWidth: 0.5))
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
                            style: StrokeStyle(lineWidth: 3, lineJoin: .round, dash: inside ? [] : [12, 8]))
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

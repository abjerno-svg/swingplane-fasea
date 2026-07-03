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
        clips = result.sorted { $0.date > $1.date }
    }

    func delete(_ clip: Clip) {
        try? FileManager.default.removeItem(at: clip.url)
        try? FileManager.default.removeItem(at: PoseAnalyzer.cacheURL(for: clip.url))
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

struct ContentView: View {
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

            HStack {
                AudioMeter(level: camera.audioLevel, impact: camera.impactFlash)
                    .frame(width: 8).padding(.leading, 6)
                Spacer()
            }.ignoresSafeArea(edges: .vertical)

            VStack(spacing: 8) {
                HStack {
                    Button { showLibrary = true } label: {
                        Image(systemName: "square.stack.3d.up.fill")
                            .font(.system(size: 20, weight: .semibold)).foregroundStyle(.white)
                            .padding(12).background(.black.opacity(0.55)).clipShape(Circle())
                    }
                    Spacer()
                    Toggle(isOn: $camera.autoMode) { Text("Auto").font(.caption).foregroundStyle(.white) }
                        .toggleStyle(.button).tint(.green)
                }
                badge(camera.statusText)
                badge(camera.poseInfo)
                FramingChecklist(camera: camera).padding(.top, 4)

                Spacer()

                Picker("Vinkel", selection: $camera.selectedView) {
                    ForEach(SwingView.allCases, id: \.self) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.segmented).frame(width: 220).padding(.bottom, 6)

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
                        .disabled(!camera.permissionGranted || camera.isRecording)
                        .padding(.trailing, 32)
                    }
                }.padding(.bottom, 40)
            }
            .padding([.horizontal, .top], 12)
        }
        .onAppear {
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

// MARK: - Bibliotek

struct LibraryView: View {
    @ObservedObject var store: ClipStore
    @Environment(\.dismiss) private var dismiss
    @State private var filter: SwingView? = nil
    @State private var showSync = false

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

                Button {
                    showSync = true
                } label: {
                    Label("Sammenlign DTL / Face-on", systemImage: "rectangle.split.2x1")
                }
                .padding(.bottom, 8)

                if filtered.isEmpty {
                    Spacer(); Text("Ingen sving endnu").foregroundStyle(.secondary); Spacer()
                } else {
                    List {
                        ForEach(filtered) { clip in
                            NavigationLink { ClipPlayerView(clip: clip) } label: { ClipRow(clip: clip) }
                        }
                        .onDelete { offsets in offsets.forEach { store.delete(filtered[$0]) } }
                    }
                }
            }
            .navigationTitle("Seneste sving")
            .toolbar { ToolbarItem(placement: .cancellationAction) { Button("Luk") { dismiss() } } }
            .sheet(isPresented: $showSync) { SyncReviewView(store: store) }
        }
        .onAppear { store.reload() }
    }
}

struct ClipRow: View {
    let clip: Clip
    private static let fmt: DateFormatter = {
        let f = DateFormatter(); f.dateFormat = "d. MMM HH:mm"; return f
    }()
    static func dateString(_ d: Date) -> String { fmt.string(from: d) }
    var body: some View {
        HStack {
            Image(systemName: clip.view == .dtl ? "figure.golf" : "person.fill").foregroundStyle(.green)
            VStack(alignment: .leading) {
                Text(clip.view.rawValue).font(.body.weight(.semibold))
                Text(Self.fmt.string(from: clip.date)).font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
            Image(systemName: "play.circle").foregroundStyle(.secondary)
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

    private var dtlClips: [Clip] { store.clips.filter { $0.view == .dtl } }
    private var faceClips: [Clip] { store.clips.filter { $0.view == .faceOn } }

    var body: some View {
        NavigationStack {
            VStack {
                HStack(spacing: 4) {
                    clipColumn(title: "Face-on", clips: faceClips, selection: $face, player: playerB)
                    clipColumn(title: "DTL", clips: dtlClips, selection: $dtl, player: playerA)
                }

                Button {
                    playSynced()
                } label: {
                    Label("Afspil synkront (justeret på impact)", systemImage: "play.rectangle.on.rectangle")
                }
                .buttonStyle(.borderedProminent)
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

    /// Sæt begge klip 1.5s før deres impact og afspil samtidig → de rammer bolden synkront.
    private func playSynced() {
        let lead = 1.5
        func seek(_ player: AVPlayer, _ clip: Clip?) {
            let impact = clip?.impact ?? lead
            let start = max(0, impact - lead)
            player.seek(to: CMTime(seconds: start, preferredTimescale: 600))
        }
        seek(playerA, dtl); seek(playerB, face)
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
    @Published var autoMode = false

    @Published var pose: PoseDict = [:]
    @Published var poseInfo = "Pose: —"

    @Published var isLevel = false
    @Published var bodyInFrame = false
    @Published var distanceOK = false
    @Published var distanceHint = ""
    @Published var clubHeadroom = false
    @Published var detectedAngle = "—"
    @Published var angleMatches = false
    @Published var selectedView: SwingView = .faceOn { didSet { recomputeFraming() } }
    var framingReady: Bool { isLevel && bodyInFrame && distanceOK && clubHeadroom && angleMatches }

    @Published var audioLevel: Float = 0
    @Published var impactFlash = false

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

    // MARK: Optagelse (manuel + auto)
    func toggleRecording() {
        let view = selectedView
        let auto = autoMode
        sessionQueue.async {
            if self.movieOutput.isRecording {
                self.movieOutput.stopRecording()
            } else {
                self.recordingView = view
                self.recordingIsAuto = auto
                self.impactSeconds = nil
                self.disableMirrorOnMovie()
                let url = FileManager.default.temporaryDirectory
                    .appendingPathComponent("rec_\(Int(Date().timeIntervalSince1970)).mov")
                self.movieOutput.startRecording(to: url, recordingDelegate: self)
                if auto { self.setStatus("Auto: klar – sving når du er klar…") }
            }
        }
    }

    /// Kaldes fra audio-tråden når impact høres i auto-tilstand.
    private func handleAutoImpact() {
        sessionQueue.async {
            guard self.recordingIsAuto, self.movieOutput.isRecording, self.impactSeconds == nil else { return }
            self.impactSeconds = CMTimeGetSeconds(self.movieOutput.recordedDuration)
            self.setStatus("Impact! trimmer sving…")
            // optag lidt efter impact, stop så
            self.sessionQueue.asyncAfter(deadline: .now() + 1.2) {
                if self.movieOutput.isRecording { self.movieOutput.stopRecording() }
            }
        }
    }

    // MARK: Framing
    func recomputeFraming() {
        guard !pose.isEmpty else {
            bodyInFrame = false; distanceOK = false; distanceHint = ""; clubHeadroom = false; angleMatches = false; return
        }
        let xs = pose.values.map { $0.x }, ys = pose.values.map { $0.y }
        guard let minX = xs.min(), let maxX = xs.max(), let minY = ys.min(), let maxY = ys.max() else { return }
        let hasHead = pose[.nose] != nil || pose[.neck] != nil
        let hasFeet = pose[.leftAnkle] != nil || pose[.rightAnkle] != nil
        bodyInFrame = hasHead && hasFeet && minX > 0.03 && maxX < 0.97 && minY > 0.02 && maxY < 0.98
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
        if peak > 0.5 { handleAutoImpact() }
        let now = CFAbsoluteTimeGetCurrent()
        guard now - lastAudioUI > 0.05 else { return }
        lastAudioUI = now
        Task { @MainActor in
            self.audioLevel = peak
            if peak > 0.5 {
                self.impactFlash = true
                Task { try? await Task.sleep(nanoseconds: 150_000_000); await MainActor.run { self.impactFlash = false } }
            }
        }
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
            trimAndSave(src: outputFileURL, view: view, impact: impact)
        } else if recordingIsAuto {
            // Auto men ingen impact hørt → kassér
            try? FileManager.default.removeItem(at: outputFileURL)
            setStatus("Auto: intet impact hørt – prøv igen")
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
        let start = max(0, impact - 2.0)
        let end = min(total, impact + 1.2)
        let impactMs = Int((impact - start) * 1000)
        let dest = ClipStore.newURL(view: view, impactMs: impactMs)

        guard let export = AVAssetExportSession(asset: asset, presetName: AVAssetExportPresetPassthrough) else {
            try? FileManager.default.moveItem(at: src, to: dest)
            Task { @MainActor in self.statusText = "Gemt ✓"; self.onClipSaved?() }
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
        }
    }
}

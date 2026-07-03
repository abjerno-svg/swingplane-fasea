//
//  ContentView.swift
//  SwingPlane – Fase A (kamera + pose + framing + klip-bibliotek)
//
//  ALT i én fil (pipeline-fil). Build 4-batch:
//   - App-privat lager (klip i Documents/clips, IKKE kamerarullen), tagget med vinkel+dato
//   - "Seneste sving"-bibliotek: filter Face-on/DTL/Begge + in-app afspilning
//   - "Vis skelet" i afspilleren: Vision-pose på det aktuelle frame (validering, pause+se)
//   - Spejlvendings-fix: isVideoMirrored=false lige før optagelse
//   - Afstands-rekalibrering (krop 42-72% af højden) + club-headroom-gate
//   + tidligere: front-default+flip, høj-fps-format, live pose-overlay, vater,
//     krop-i-frame, vinkel-aflæsning, lyd-meter/impact.
//
//  TUNING-PUNKTER (device, ikke compile): pose-orientering (visionOrientation),
//  lyd-format (audioPeak), framing-tærskler.
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

// MARK: - Klip-model + lager (app-privat)

struct Clip: Identifiable {
    let id = UUID()
    let url: URL
    let view: SwingView
    let date: Date
}

final class ClipStore: ObservableObject {
    @Published var clips: [Clip] = []

    static let directory: URL = {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let dir = docs.appendingPathComponent("clips", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }()

    static func newURL(view: SwingView) -> URL {
        let ts = Int(Date().timeIntervalSince1970)
        return directory.appendingPathComponent("swing_\(ts)_\(view.fileKey).mov")
    }

    func reload() {
        let files = (try? FileManager.default.contentsOfDirectory(
            at: Self.directory, includingPropertiesForKeys: nil)) ?? []
        var result: [Clip] = []
        for url in files where url.pathExtension == "mov" {
            let parts = url.deletingPathExtension().lastPathComponent.split(separator: "_")
            guard parts.count >= 3, let ts = Double(parts[1]) else { continue }
            result.append(Clip(url: url,
                               view: SwingView.fromFileKey(String(parts[2])),
                               date: Date(timeIntervalSince1970: ts)))
        }
        clips = result.sorted { $0.date > $1.date }
    }

    func delete(_ clip: Clip) {
        try? FileManager.default.removeItem(at: clip.url)
        reload()
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
                            .font(.system(size: 20, weight: .semibold))
                            .foregroundStyle(.white)
                            .padding(12).background(.black.opacity(0.55)).clipShape(Circle())
                    }
                    Spacer()
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
                                .font(.system(size: 24, weight: .semibold))
                                .foregroundStyle(.white)
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
            camera.start()
            store.reload()
        }
        .sheet(isPresented: $showLibrary) { LibraryView(store: store) }
    }

    private func badge(_ text: String) -> some View {
        Text(text)
            .font(.caption.weight(.semibold))
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

                if filtered.isEmpty {
                    Spacer()
                    Text("Ingen sving endnu").foregroundStyle(.secondary)
                    Spacer()
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
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Luk") { dismiss() }
                }
            }
        }
        .onAppear { store.reload() }
    }
}

struct ClipRow: View {
    let clip: Clip
    private static let fmt: DateFormatter = {
        let f = DateFormatter(); f.dateFormat = "d. MMM HH:mm"; return f
    }()
    var body: some View {
        HStack {
            Image(systemName: clip.view == .dtl ? "figure.golf" : "person.fill")
                .foregroundStyle(.green)
            VStack(alignment: .leading) {
                Text(clip.view.rawValue).font(.body.weight(.semibold))
                Text(Self.fmt.string(from: clip.date)).font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
            Image(systemName: "play.circle").foregroundStyle(.secondary)
        }
    }
}

// MARK: - Afspiller + skelet-validering

struct ClipPlayerView: View {
    let clip: Clip
    @State private var player = AVPlayer()
    @State private var showSkeleton = false
    @State private var pose: PoseDict = [:]
    @State private var videoSize: CGSize = .zero

    var body: some View {
        VStack {
            ZStack {
                VideoPlayer(player: player)
                if showSkeleton {
                    PoseOverlay(pose: pose, videoSize: videoSize, fill: false)
                        .allowsHitTesting(false)
                }
            }
            .frame(maxHeight: 480)

            Button(showSkeleton ? "Skjul skelet" : "Vis skelet på dette frame") {
                if showSkeleton { showSkeleton = false } else { analyzeFrame() }
            }
            .buttonStyle(.borderedProminent)
            .padding()

            Text("Pause videoen på det ønskede frame og tryk for at validere pose.")
                .font(.caption).foregroundStyle(.secondary).multilineTextAlignment(.center)
                .padding(.horizontal)
            Spacer()
        }
        .navigationTitle(clip.view.rawValue)
        .navigationBarTitleDisplayMode(.inline)
        .onAppear { player.replaceCurrentItem(with: AVPlayerItem(url: clip.url)) }
        .onDisappear { player.pause() }
    }

    private func analyzeFrame() {
        player.pause()
        let asset = AVURLAsset(url: clip.url)
        let gen = AVAssetImageGenerator(asset: asset)
        gen.appliesPreferredTrackTransform = true
        gen.requestedTimeToleranceBefore = .zero
        gen.requestedTimeToleranceAfter = .zero
        gen.generateCGImageAsynchronously(for: player.currentTime()) { cg, _, _ in
            guard let cg else { return }
            let handler = VNImageRequestHandler(cgImage: cg, orientation: .up, options: [:])
            let req = VNDetectHumanBodyPoseRequest()
            try? handler.perform([req])
            var result: PoseDict = [:]
            if let obs = req.results?.first, let pts = try? obs.recognizedPoints(.all) {
                for (name, p) in pts where p.confidence > 0.3 {
                    result[name] = CGPoint(x: p.location.x, y: 1 - p.location.y)
                }
            }
            let size = CGSize(width: cg.width, height: cg.height)
            DispatchQueue.main.async {
                self.pose = result
                self.videoSize = size
                self.showSkeleton = true
            }
        }
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
        }
        .clipShape(Capsule())
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
            if camera.framingReady {
                Text("KLAR").font(.caption.bold()).foregroundStyle(.green)
            }
        }
        .padding(.horizontal, 12).padding(.vertical, 8)
        .background(.black.opacity(0.5)).clipShape(RoundedRectangle(cornerRadius: 10))
    }
    private func row(_ text: String, _ ok: Bool) -> some View {
        HStack(spacing: 6) {
            Image(systemName: ok ? "checkmark.circle.fill" : "xmark.circle")
                .foregroundStyle(ok ? .green : .red)
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

    private var startPosition: AVCaptureDevice.Position { isFrontCamera ? .front : .back }
    private var visionOrientation: CGImagePropertyOrientation { isFrontCamera ? .leftMirrored : .right }

    static let bones: [(VNHumanBodyPoseObservation.JointName, VNHumanBodyPoseObservation.JointName)] = [
        (.neck, .nose),
        (.neck, .leftShoulder), (.neck, .rightShoulder),
        (.leftShoulder, .leftElbow), (.leftElbow, .leftWrist),
        (.rightShoulder, .rightElbow), (.rightElbow, .rightWrist),
        (.leftShoulder, .leftHip), (.rightShoulder, .rightHip),
        (.leftHip, .rightHip),
        (.leftHip, .leftKnee), (.leftKnee, .leftAnkle),
        (.rightHip, .rightKnee), (.rightKnee, .rightAnkle)
    ]

    // MARK: Start
    func start() {
        Task {
            let cam = await AVCaptureDevice.requestAccess(for: .video)
            let mic = await AVCaptureDevice.requestAccess(for: .audio)
            await MainActor.run { self.permissionGranted = cam && mic }
            if cam && mic {
                sessionQueue.async { self.configureSession() }
                startMotion()
            } else {
                setStatus("Kamera-/mikrofon-adgang nægtet")
            }
        }
    }

    // MARK: Konfiguration
    private func configureSession() {
        session.beginConfiguration()
        guard addVideoInput(position: startPosition) else {
            session.commitConfiguration(); setStatus("Kunne ikke tilgå kameraet"); return
        }
        if let audioDevice = AVCaptureDevice.default(for: .audio),
           let audioInput = try? AVCaptureDeviceInput(device: audioDevice),
           session.canAddInput(audioInput) {
            session.addInput(audioInput)
        }
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
        session.addInput(input)
        videoDeviceInput = input
        return true
    }

    /// Sæt "ingen spejling" på den LIVE movie-forbindelse (kaldes lige før optagelse,
    /// hvor forbindelsen findes — ellers rammer det ingenting).
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
            } else {
                self.session.addInput(current); self.videoDeviceInput = current
            }
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
                self.activeFPS = fps
                self.activeResolution = "\(dims.width)×\(dims.height)"
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

    // MARK: Optagelse
    func toggleRecording() {
        let view = selectedView
        sessionQueue.async {
            if self.movieOutput.isRecording {
                self.movieOutput.stopRecording()
            } else {
                self.recordingView = view
                self.disableMirrorOnMovie()      // live-forbindelse findes nu
                let url = FileManager.default.temporaryDirectory
                    .appendingPathComponent("rec_\(Int(Date().timeIntervalSince1970)).mov")
                self.movieOutput.startRecording(to: url, recordingDelegate: self)
            }
        }
    }

    // MARK: Framing
    func recomputeFraming() {
        guard !pose.isEmpty else {
            bodyInFrame = false; distanceOK = false; distanceHint = ""; clubHeadroom = false; angleMatches = false
            return
        }
        let xs = pose.values.map { $0.x }, ys = pose.values.map { $0.y }
        guard let minX = xs.min(), let maxX = xs.max(),
              let minY = ys.min(), let maxY = ys.max() else { return }

        let hasHead = pose[.nose] != nil || pose[.neck] != nil
        let hasFeet = pose[.leftAnkle] != nil || pose[.rightAnkle] != nil
        bodyInFrame = hasHead && hasFeet && minX > 0.03 && maxX < 0.97 && minY > 0.02 && maxY < 0.98

        let height = maxY - minY
        if height < 0.42 { distanceOK = false; distanceHint = "Træd tættere på" }
        else if height > 0.72 { distanceOK = false; distanceHint = "Træd tilbage" }
        else { distanceOK = true; distanceHint = "" }

        clubHeadroom = minY > 0.12          // luft over hovedet til køllen i toppen

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
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer,
                                            orientation: visionOrientation, options: [:])
        do {
            try handler.perform([poseRequest])
            guard let obs = poseRequest.results?.first else {
                Task { @MainActor in self.pose = [:]; self.poseInfo = "Pose: ingen person"; self.recomputeFraming() }
                return
            }
            let pts = (try? obs.recognizedPoints(.all)) ?? [:]
            var result: PoseDict = [:]
            var confSum: Float = 0, count = 0
            for (name, pt) in pts where pt.confidence > 0.3 {
                result[name] = CGPoint(x: pt.location.x, y: 1 - pt.location.y)
                confSum += pt.confidence; count += 1
            }
            let avg = count > 0 ? confSum / Float(count) : 0
            Task { @MainActor in
                self.pose = result
                self.poseInfo = count > 0
                    ? "Pose: \(count)/19 led · konf \(String(format: "%.2f", avg))"
                    : "Pose: ingen person"
                self.recomputeFraming()
            }
        } catch { }
    }

    private func processAudio(_ sampleBuffer: CMSampleBuffer) {
        let peak = audioPeak(sampleBuffer)
        let now = CFAbsoluteTimeGetCurrent()
        guard now - lastAudioUI > 0.05 else { return }
        lastAudioUI = now
        Task { @MainActor in
            self.audioLevel = peak
            if peak > 0.5 {
                self.impactFlash = true
                Task { try? await Task.sleep(nanoseconds: 150_000_000)
                       await MainActor.run { self.impactFlash = false } }
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

// MARK: - Optage-delegate (gem app-privat)

extension CameraManager: AVCaptureFileOutputRecordingDelegate {
    func fileOutput(_ output: AVCaptureFileOutput, didStartRecordingTo fileURL: URL,
                    from connections: [AVCaptureConnection]) {
        Task { @MainActor in self.isRecording = true; self.statusText = "Optager…" }
    }

    func fileOutput(_ output: AVCaptureFileOutput, didFinishRecordingTo outputFileURL: URL,
                    from connections: [AVCaptureConnection], error: Error?) {
        let view = recordingView
        let dest = ClipStore.newURL(view: view)
        do {
            try FileManager.default.moveItem(at: outputFileURL, to: dest)
            Task { @MainActor in
                self.isRecording = false
                self.statusText = "Gemt ✓ (\(view.rawValue), \(Int(self.activeFPS)) fps)"
                self.onClipSaved?()
            }
        } catch {
            Task { @MainActor in
                self.isRecording = false
                self.statusText = "Fejl ved gem: \(error.localizedDescription)"
            }
        }
    }
}

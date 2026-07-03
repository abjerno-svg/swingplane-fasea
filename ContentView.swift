//
//  ContentView.swift
//  SwingPlane – Fase A (de-risk: kamera-capture + pose + framing-fundament)
//
//  ALT i én fil, så den kan erstatte den auto-genererede ContentView.swift
//  direkte (ingen nye filer at tilføje i Xcode).
//
//  Build 3-batch:
//   - Front-kamera default + flip (bag = 240fps, front = 120fps)
//   - Bedste høj-fps-format, ingen spejling i gemt fil
//   - Vision 2D-pose-overlay + aflæsning (antal led/konfidens)
//   - Vater-indikator (CoreMotion)
//   - Krop-i-frame + afstands-guide (fra pose)
//   - Vinkel-aflæsning DTL/Face-on/Diagonal (skulder-linje)
//   - Lyd-niveau-meter + impact-blink
//   - DTL/Face-on-vælger + samlet framing-tjekliste
//
//  TUNING-PUNKTER (forventet finjustering PÅ enheden, ikke compile):
//   1) Pose-orientering/spejling (visionOrientation) — hvis skelettet står
//      roteret/spejlet, er det her knappen sidder.
//   2) Lyd-format i audioPeak() — håndterer Float32 + Int16; hvis meteret er dødt
//      er det formatet.
//   3) Framing-tærskler (distance/vinkel) er første-gæt, kalibreres mod rigtige klip.
//

import SwiftUI
import Combine
import AVFoundation
import Photos
import Vision
import CoreMotion

typealias PoseDict = [VNHumanBodyPoseObservation.JointName: CGPoint]

enum SwingView: String, CaseIterable {
    case faceOn = "Face-on"
    case dtl = "DTL"
}

// MARK: - View

struct ContentView: View {
    @StateObject private var camera = CameraManager()

    var body: some View {
        ZStack {
            // Live-preview
            if camera.permissionGranted {
                CameraPreview(session: camera.session).ignoresSafeArea()
            } else {
                Color.black.ignoresSafeArea()
            }

            // Pose-skelet-overlay
            PoseOverlay(pose: camera.pose, videoSize: camera.videoPortraitSize)
                .allowsHitTesting(false)
                .ignoresSafeArea()

            // Lyd-meter (venstre kant)
            HStack {
                AudioMeter(level: camera.audioLevel, impact: camera.impactFlash)
                    .frame(width: 8)
                    .padding(.leading, 6)
                Spacer()
            }
            .ignoresSafeArea(edges: .vertical)

            VStack(spacing: 8) {
                // Kamera + fps
                badge(camera.statusText)
                // Pose-aflæsning
                badge(camera.poseInfo)

                // Framing-tjekliste
                FramingChecklist(camera: camera)
                    .padding(.top, 4)

                Spacer()

                // DTL / Face-on-vælger
                Picker("Vinkel", selection: $camera.selectedView) {
                    ForEach(SwingView.allCases, id: \.self) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.segmented)
                .frame(width: 220)
                .padding(.bottom, 6)

                // Bund-kontroller: optageknap centreret, flip til højre
                ZStack {
                    Button(action: { camera.toggleRecording() }) {
                        ZStack {
                            Circle().stroke(.white, lineWidth: 4).frame(width: 84, height: 84)
                            Circle().fill(camera.isRecording ? Color.red : Color.white)
                                .frame(width: 70, height: 70)
                        }
                    }
                    .disabled(!camera.permissionGranted)

                    HStack {
                        Spacer()
                        Button(action: { camera.flipCamera() }) {
                            Image(systemName: "arrow.triangle.2.circlepath.camera")
                                .font(.system(size: 24, weight: .semibold))
                                .foregroundStyle(.white)
                                .padding(16)
                                .background(.black.opacity(0.55))
                                .clipShape(Circle())
                        }
                        .disabled(!camera.permissionGranted || camera.isRecording)
                        .padding(.trailing, 32)
                    }
                }
                .padding(.bottom, 40)
            }
            .padding(.top, 12)
        }
        .onAppear { camera.start() }
    }

    private func badge(_ text: String) -> some View {
        Text(text)
            .font(.caption.weight(.semibold))
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(.black.opacity(0.55))
            .foregroundStyle(.white)
            .clipShape(Capsule())
    }
}

#Preview { ContentView() }

// MARK: - Pose-overlay (Canvas)

struct PoseOverlay: View {
    let pose: PoseDict
    let videoSize: CGSize

    var body: some View {
        Canvas { ctx, size in
            guard videoSize.width > 0, videoSize.height > 0, !pose.isEmpty else { return }
            // aspectFill-mapping fra portræt-normaliserede punkter til view
            let scale = max(size.width / videoSize.width, size.height / videoSize.height)
            let dispW = videoSize.width * scale
            let dispH = videoSize.height * scale
            let ox = (size.width - dispW) / 2
            let oy = (size.height - dispH) / 2
            func p(_ n: CGPoint) -> CGPoint {
                CGPoint(x: ox + n.x * dispW, y: oy + n.y * dispH)
            }
            // knogler
            for (a, b) in CameraManager.bones {
                if let pa = pose[a], let pb = pose[b] {
                    var path = Path()
                    path.move(to: p(pa)); path.addLine(to: p(pb))
                    ctx.stroke(path, with: .color(.green.opacity(0.9)), lineWidth: 3)
                }
            }
            // led
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
    let level: Float      // 0..1
    let impact: Bool

    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .bottom) {
                Capsule().fill(.black.opacity(0.4))
                Capsule()
                    .fill(impact ? Color.red : Color.green)
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
            row("Vinkel: \(camera.detectedAngle)", camera.angleMatches)
            if camera.framingReady {
                Text("KLAR").font(.caption.bold()).foregroundStyle(.green)
            }
        }
        .padding(.horizontal, 12).padding(.vertical, 8)
        .background(.black.opacity(0.5))
        .clipShape(RoundedRectangle(cornerRadius: 10))
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

    // Kamera-tilstand
    @Published var isRecording = false
    @Published var permissionGranted = false
    @Published var statusText = "Anmoder om adgang…"
    @Published var activeFPS: Double = 0
    @Published var activeResolution = ""
    @Published var isFrontCamera = true
    @Published var videoPortraitSize = CGSize(width: 1080, height: 1920)

    // Pose
    @Published var pose: PoseDict = [:]
    @Published var poseInfo = "Pose: —"

    // Framing-gates
    @Published var isLevel = false
    @Published var bodyInFrame = false
    @Published var distanceOK = false
    @Published var distanceHint = ""
    @Published var detectedAngle = "—"
    @Published var angleMatches = false
    @Published var selectedView: SwingView = .faceOn { didSet { recomputeFraming() } }
    var framingReady: Bool { isLevel && bodyInFrame && distanceOK && angleMatches }

    // Lyd
    @Published var audioLevel: Float = 0
    @Published var impactFlash = false

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

        applyNoMirroring()
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

    private func applyNoMirroring() {
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
                self.applyNoMirroring()
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

    // MARK: Vater (CoreMotion)
    private func startMotion() {
        guard motion.isDeviceMotionAvailable else { return }
        motion.deviceMotionUpdateInterval = 0.1
        motion.startDeviceMotionUpdates(to: .main) { [weak self] m, _ in
            guard let self, let m else { return }
            // Roll = venstre/højre-hældning når telefonen står lodret i portræt.
            let roll = atan2(m.gravity.x, -m.gravity.y) * 180 / .pi
            self.isLevel = abs(roll) < 4          // ±4° tolerance (tuning)
            self.recomputeFraming()
        }
    }

    // MARK: Optagelse
    func toggleRecording() {
        sessionQueue.async {
            if self.movieOutput.isRecording {
                self.movieOutput.stopRecording()
            } else {
                let url = FileManager.default.temporaryDirectory
                    .appendingPathComponent("swing_\(Int(Date().timeIntervalSince1970)).mov")
                self.movieOutput.startRecording(to: url, recordingDelegate: self)
            }
        }
    }

    // MARK: Framing-beregning (kaldes efter pose/motion — altid på main-tråden)
    func recomputeFraming() {
        guard !pose.isEmpty else {
            bodyInFrame = false; distanceOK = false; distanceHint = ""; angleMatches = false
            return
        }
        let xs = pose.values.map { $0.x }, ys = pose.values.map { $0.y }
        guard let minX = xs.min(), let maxX = xs.max(),
              let minY = ys.min(), let maxY = ys.max() else { return }

        let hasHead = pose[.nose] != nil || pose[.neck] != nil
        let hasFeet = pose[.leftAnkle] != nil || pose[.rightAnkle] != nil
        bodyInFrame = hasHead && hasFeet &&
            minX > 0.03 && maxX < 0.97 && minY > 0.02 && maxY < 0.98

        let height = maxY - minY
        if height < 0.55 { distanceOK = false; distanceHint = "Træd tættere på" }
        else if height > 0.92 { distanceOK = false; distanceHint = "Træd tilbage" }
        else { distanceOK = true; distanceHint = "" }

        // Vinkel fra skulderbredde relativt til kropshøjde
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

// MARK: - Sample-buffer-delegate (video-pose + lyd)

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate,
                         AVCaptureAudioDataOutputSampleBufferDelegate {

    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        if output === videoDataOutput {
            processVideo(sampleBuffer)
        } else if output === audioDataOutput {
            processAudio(sampleBuffer)
        }
    }

    private func processVideo(_ sampleBuffer: CMSampleBuffer) {
        let now = CFAbsoluteTimeGetCurrent()
        guard now - lastPoseTime > 0.04 else { return }   // ~25 Hz
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
                result[name] = CGPoint(x: pt.location.x, y: 1 - pt.location.y)  // → top-venstre
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
        } catch {
            // stille – pose fejler bare på enkeltframes
        }
    }

    private func processAudio(_ sampleBuffer: CMSampleBuffer) {
        let peak = audioPeak(sampleBuffer)
        let now = CFAbsoluteTimeGetCurrent()
        guard now - lastAudioUI > 0.05 else { return }     // ~20 Hz UI
        lastAudioUI = now
        Task { @MainActor in
            self.audioLevel = peak
            if peak > 0.5 {                                // impact-tærskel (tuning)
                self.impactFlash = true
                Task {
                    try? await Task.sleep(nanoseconds: 150_000_000)
                    await MainActor.run { self.impactFlash = false }
                }
            }
        }
    }

    /// Peak-amplitude fra en lyd-buffer. Håndterer Float32 + Int16.
    private func audioPeak(_ sampleBuffer: CMSampleBuffer) -> Float {
        guard let fd = CMSampleBufferGetFormatDescription(sampleBuffer),
              let asbdPtr = CMAudioFormatDescriptionGetStreamBasicDescription(fd) else { return 0 }
        let asbd = asbdPtr.pointee
        let isFloat = (asbd.mFormatFlags & kAudioFormatFlagIsFloat) != 0
        let bytesPerChannel = Int(asbd.mBitsPerChannel / 8)

        var blockBuffer: CMBlockBuffer?
        var abl = AudioBufferList()
        let status = CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(
            sampleBuffer,
            bufferListSizeNeededOut: nil,
            bufferListOut: &abl,
            bufferListSize: MemoryLayout<AudioBufferList>.size,
            blockBufferAllocator: nil,
            blockBufferMemoryAllocator: nil,
            flags: [],
            blockBufferOut: &blockBuffer)
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

// MARK: - Optage-delegate (fil)

extension CameraManager: AVCaptureFileOutputRecordingDelegate {
    func fileOutput(_ output: AVCaptureFileOutput, didStartRecordingTo fileURL: URL,
                    from connections: [AVCaptureConnection]) {
        Task { @MainActor in self.isRecording = true; self.statusText = "Optager…" }
    }

    func fileOutput(_ output: AVCaptureFileOutput, didFinishRecordingTo outputFileURL: URL,
                    from connections: [AVCaptureConnection], error: Error?) {
        Task { @MainActor in self.isRecording = false }
        // Gemmes i Fotos under de-risk (kan granskes bagefter). Skiftes til app-privat
        // lager + auto-trim når rolling-buffer bygges (beslutning 2026-07-03).
        PHPhotoLibrary.requestAuthorization(for: .addOnly) { status in
            guard status == .authorized else {
                self.setStatus("Foto-adgang nægtet – klip i temp-mappe"); return
            }
            PHPhotoLibrary.shared().performChanges {
                PHAssetCreationRequest.forAsset().addResource(with: .video, fileURL: outputFileURL, options: nil)
            } completionHandler: { success, err in
                if success { self.setStatus("Gemt i Fotos ✓ (\(Int(self.activeFPS)) fps @ \(self.activeResolution))") }
                else { self.setStatus("Fejl ved gem: \(err?.localizedDescription ?? "ukendt")") }
            }
        }
    }
}

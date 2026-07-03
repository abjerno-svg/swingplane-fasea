//
//  ContentView.swift
//  SwingPlane – Fase A (de-risk: kamera-capture)
//
//  ALT i én fil, så den kan erstatte den auto-genererede ContentView.swift
//  direkte (ingen nye filer at tilføje i Xcode).
//
//  Optager i højest mulige fps med synkron lyd, gemmer klippet i Fotos, og
//  viser den FAKTISKE fps + opløsning + hvilket kamera, så vi kan bekræfte
//  hardwaren live på enheden.
//
//  KAMERA-VALG: FRONT (selfie) er default — så man kan se skærmen/framing
//  mens man filmer sig selv solo. Frontkameraet topper typisk ved 1080p@120
//  (bagkameraet kan 240). Flip-knappen skifter til bagkameraet når man vil
//  have maks kvalitet.
//

import SwiftUI
import Combine
import AVFoundation
import Photos

// MARK: - View

struct ContentView: View {
    @StateObject private var camera = CameraManager()

    var body: some View {
        ZStack {
            if camera.permissionGranted {
                CameraPreview(session: camera.session)
                    .ignoresSafeArea()
            } else {
                Color.black.ignoresSafeArea()
            }

            VStack {
                // Status-badge: bekræfter kamera + fps/opløsning live
                Text(camera.statusText)
                    .font(.callout.weight(.semibold))
                    .padding(.horizontal, 14)
                    .padding(.vertical, 8)
                    .background(.black.opacity(0.55))
                    .foregroundStyle(.white)
                    .clipShape(Capsule())
                    .padding(.top, 12)

                Spacer()

                // Bund-kontroller: optageknap centreret, flip-knap til højre
                ZStack {
                    // Optageknap (rød mens der optages)
                    Button(action: { camera.toggleRecording() }) {
                        ZStack {
                            Circle().stroke(.white, lineWidth: 4).frame(width: 84, height: 84)
                            Circle().fill(camera.isRecording ? Color.red : Color.white)
                                .frame(width: 70, height: 70)
                        }
                    }
                    .disabled(!camera.permissionGranted)

                    // Flip-knap (skift front/bag) — deaktiveret under optagelse
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
                .padding(.bottom, 44)
            }
        }
        .onAppear { camera.start() }
    }
}

#Preview {
    ContentView()
}

// MARK: - Kamera-preview (SwiftUI-bro til AVCaptureVideoPreviewLayer)

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
    @Published var isFrontCamera = true          // FRONT = default (solo-brug)

    let session = AVCaptureSession()

    private let movieOutput = AVCaptureMovieFileOutput()
    private let sessionQueue = DispatchQueue(label: "swingplane.camera.session")
    private var videoDeviceInput: AVCaptureDeviceInput?

    private var startPosition: AVCaptureDevice.Position { isFrontCamera ? .front : .back }

    // MARK: - Start

    func start() {
        Task {
            let cam = await AVCaptureDevice.requestAccess(for: .video)
            let mic = await AVCaptureDevice.requestAccess(for: .audio)
            await MainActor.run { self.permissionGranted = cam && mic }
            if cam && mic {
                sessionQueue.async { self.configureSession() }
            } else {
                setStatus("Kamera-/mikrofon-adgang nægtet")
            }
        }
    }

    // MARK: - Konfiguration

    private func configureSession() {
        session.beginConfiguration()

        // 1) Video-input: front som default
        guard addVideoInput(position: startPosition) else {
            session.commitConfiguration()
            setStatus("Kunne ikke tilgå kameraet")
            return
        }

        // 2) Audio-input (impact-lyd, synkron med video)
        if let audioDevice = AVCaptureDevice.default(for: .audio),
           let audioInput = try? AVCaptureDeviceInput(device: audioDevice),
           session.canAddInput(audioInput) {
            session.addInput(audioInput)
        }

        // 3) Movie-output (video + lyd i én fil)
        if session.canAddOutput(movieOutput) {
            session.addOutput(movieOutput)
        }
        applyNoMirroring()

        session.commitConfiguration()

        if let device = videoDeviceInput?.device {
            configureHighFrameRate(device: device)
        }
        session.startRunning()
    }

    /// Tilføjer video-input for en given kamera-position og husker det.
    private func addVideoInput(position: AVCaptureDevice.Position) -> Bool {
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: position),
              let input = try? AVCaptureDeviceInput(device: device),
              session.canAddInput(input) else {
            return false
        }
        session.addInput(input)
        videoDeviceInput = input
        return true
    }

    /// Ingen spejling i den GEMTE fil — bevar sand venstre/højre-geometri
    /// (vigtigt for pose- og klubhoved-analyse på front-optagelser).
    private func applyNoMirroring() {
        if let c = movieOutput.connection(with: .video), c.isVideoMirroringSupported {
            c.automaticallyAdjustsVideoMirroring = false
            c.isVideoMirrored = false
        }
    }

    // MARK: - Kamera-skift (front/bag)

    func flipCamera() {
        sessionQueue.async {
            guard let current = self.videoDeviceInput else { return }
            let newPosition: AVCaptureDevice.Position = (current.device.position == .front) ? .back : .front

            self.session.beginConfiguration()
            self.session.removeInput(current)

            if self.addVideoInput(position: newPosition) {
                self.applyNoMirroring()
                Task { @MainActor in self.isFrontCamera = (newPosition == .front) }
            } else {
                // Rul tilbage hvis det nye kamera ikke kunne tilføjes
                self.session.addInput(current)
                self.videoDeviceInput = current
            }
            self.session.commitConfiguration()

            if let device = self.videoDeviceInput?.device {
                self.configureHighFrameRate(device: device)
            }
        }
    }

    // MARK: - Format-valg (høj fps)

    /// Vælger bedste format: foretræk 1080p+ (skarphed), derefter højest fps,
    /// derefter størst opløsning. Låser fps fast (cap 240).
    /// → Bag: typisk 1080p@240. Front: typisk 1080p@120 (ikke 4K@60).
    private func configureHighFrameRate(device: AVCaptureDevice, cap: Double = 240) {
        func maxRate(_ f: AVCaptureDevice.Format) -> Double {
            f.videoSupportedFrameRateRanges.map(\.maxFrameRate).max() ?? 0
        }
        func rank(_ f: AVCaptureDevice.Format) -> (Int, Double, Int32) {
            let d = CMVideoFormatDescriptionGetDimensions(f.formatDescription)
            let is1080plus = d.width >= 1920 ? 1 : 0     // 1080p eller bedre foretrækkes
            return (is1080plus, maxRate(f), d.width)
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
            let duration = CMTime(value: 1, timescale: CMTimeScale(fps))
            device.activeVideoMinFrameDuration = duration
            device.activeVideoMaxFrameDuration = duration
            device.unlockForConfiguration()

            let dims = CMVideoFormatDescriptionGetDimensions(chosen.formatDescription)
            let cam = (device.position == .front) ? "Front" : "Bag"
            Task { @MainActor in
                self.activeFPS = fps
                self.activeResolution = "\(dims.width)×\(dims.height)"
                self.statusText = "\(cam) – \(Int(fps)) fps @ \(dims.width)×\(dims.height)"
            }
        } catch {
            setStatus("Kunne ikke sætte fps: \(error.localizedDescription)")
        }
    }

    // MARK: - Optagelse

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

    // MARK: - Hjælpere

    private func setStatus(_ text: String) {
        Task { @MainActor in self.statusText = text }
    }
}

// MARK: - Optage-delegate

extension CameraManager: AVCaptureFileOutputRecordingDelegate {

    func fileOutput(_ output: AVCaptureFileOutput,
                    didStartRecordingTo fileURL: URL,
                    from connections: [AVCaptureConnection]) {
        Task { @MainActor in
            self.isRecording = true
            self.statusText = "Optager…"
        }
    }

    func fileOutput(_ output: AVCaptureFileOutput,
                    didFinishRecordingTo outputFileURL: URL,
                    from connections: [AVCaptureConnection],
                    error: Error?) {
        Task { @MainActor in self.isRecording = false }

        // Gemmes i Fotos under de-risk-fasen, så skarphed/fps kan granskes bagefter.
        // (Skiftes til app-privat lager + auto-trim når rolling-buffer bygges, jf. beslutning 2026-07-03.)
        PHPhotoLibrary.requestAuthorization(for: .addOnly) { status in
            guard status == .authorized else {
                self.setStatus("Foto-adgang nægtet – klip ligger i app'ens temp-mappe")
                return
            }
            PHPhotoLibrary.shared().performChanges {
                PHAssetCreationRequest.forAsset()
                    .addResource(with: .video, fileURL: outputFileURL, options: nil)
            } completionHandler: { success, err in
                if success {
                    self.setStatus("Gemt i Fotos ✓  (\(Int(self.activeFPS)) fps @ \(self.activeResolution))")
                } else {
                    self.setStatus("Fejl ved gem: \(err?.localizedDescription ?? "ukendt")")
                }
            }
        }
    }
}

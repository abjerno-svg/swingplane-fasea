//
//  ContentView.swift
//  SwingPlane – Fase A (de-risk: kamera-capture)
//
//  ALT i én fil, så den kan erstatte den auto-genererede ContentView.swift
//  direkte (ingen nye filer at tilføje i Xcode).
//
//  Optager i højest mulige fps (helst 240) med synkron lyd, gemmer klippet
//  i Fotos, og viser den FAKTISKE fps + opløsning så vi kan bekræfte
//  hardwaren live på enheden.
//

import SwiftUI
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
                // Status-badge: bekræfter fps/opløsning live
                Text(camera.statusText)
                    .font(.callout.weight(.semibold))
                    .padding(.horizontal, 14)
                    .padding(.vertical, 8)
                    .background(.black.opacity(0.55))
                    .foregroundStyle(.white)
                    .clipShape(Capsule())
                    .padding(.top, 12)

                Spacer()

                // Optageknap (rød mens der optages)
                Button(action: { camera.toggleRecording() }) {
                    ZStack {
                        Circle().stroke(.white, lineWidth: 4).frame(width: 84, height: 84)
                        Circle().fill(camera.isRecording ? Color.red : Color.white)
                            .frame(width: 70, height: 70)
                    }
                }
                .disabled(!camera.permissionGranted)
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

    let session = AVCaptureSession()

    private let movieOutput = AVCaptureMovieFileOutput()
    private let sessionQueue = DispatchQueue(label: "swingplane.camera.session")

    // Start: bed om adgang + konfigurer
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

    private func configureSession() {
        session.beginConfiguration()

        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera,
                                                   for: .video, position: .back),
              let videoInput = try? AVCaptureDeviceInput(device: device),
              session.canAddInput(videoInput) else {
            session.commitConfiguration()
            setStatus("Kunne ikke tilgå bagkameraet")
            return
        }
        session.addInput(videoInput)

        if let audioDevice = AVCaptureDevice.default(for: .audio),
           let audioInput = try? AVCaptureDeviceInput(device: audioDevice),
           session.canAddInput(audioInput) {
            session.addInput(audioInput)
        }

        if session.canAddOutput(movieOutput) {
            session.addOutput(movieOutput)
        }

        session.commitConfiguration()
        configureHighFrameRate(device: device, desiredFPS: 240)
        session.startRunning()
    }

    private func configureHighFrameRate(device: AVCaptureDevice, desiredFPS: Double) {
        var best: (format: AVCaptureDevice.Format, fps: Double, width: Int32)?

        for format in device.formats {
            let dims = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
            let maxRate = format.videoSupportedFrameRateRanges.map(\.maxFrameRate).max() ?? 0
            let supportsDesired = maxRate >= desiredFPS

            if let cur = best {
                let curSupports = cur.fps >= desiredFPS
                if supportsDesired && !curSupports {
                    best = (format, maxRate, dims.width)
                } else if supportsDesired == curSupports,
                          dims.width > cur.width ||
                          (dims.width == cur.width && maxRate > cur.fps) {
                    best = (format, maxRate, dims.width)
                }
            } else {
                best = (format, maxRate, dims.width)
            }
        }

        guard let chosen = best else { return }

        do {
            try device.lockForConfiguration()
            device.activeFormat = chosen.format
            let fps = min(desiredFPS, chosen.fps)
            let duration = CMTime(value: 1, timescale: CMTimeScale(fps))
            device.activeVideoMinFrameDuration = duration
            device.activeVideoMaxFrameDuration = duration
            device.unlockForConfiguration()

            let dims = CMVideoFormatDescriptionGetDimensions(chosen.format.formatDescription)
            Task { @MainActor in
                self.activeFPS = fps
                self.activeResolution = "\(dims.width)×\(dims.height)"
                self.statusText = "Klar – \(Int(fps)) fps @ \(dims.width)×\(dims.height)"
            }
        } catch {
            setStatus("Kunne ikke sætte fps: \(error.localizedDescription)")
        }
    }

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

    private func setStatus(_ text: String) {
        Task { @MainActor in self.statusText = text }
    }
}

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

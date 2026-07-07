# Konverterer best_v3.onnx (YOLO klubhoved-detektor) til Core ML (SwingClub.mlpackage).
# Køres på Mac'en (SSH er nok, ingen VNC nødvendig):
#
#   mkdir -p ~/clubmodel && cd ~/clubmodel
#   curl -fsSL "https://raw.githubusercontent.com/abjerno-svg/swingplane-fasea/refs/heads/main/best_v3.onnx?t=$(date +%s)" -o best_v3.onnx
#   curl -fsSL "https://raw.githubusercontent.com/abjerno-svg/swingplane-fasea/refs/heads/main/convert_clubhead_model.py?t=$(date +%s)" -o convert_clubhead_model.py
#   python3 -m venv env && source env/bin/activate
#   pip install -q torch onnx onnx2torch coremltools onnxruntime pillow numpy
#   python3 convert_clubhead_model.py
#
# Output: SwingClub.mlpackage  →  drag ind i Xcode-projektet (target-membership ✓).
# Scriptet validerer selv Core ML-outputtet mod onnxruntime (maks-afvigelse skal være lille).

import numpy as np

ONNX = "best_v3.onnx"
OUT = "SwingClub.mlpackage"

print("1/4  Loader ONNX → PyTorch ...")
import torch
from onnx2torch import convert
tm = convert(ONNX).eval()

print("2/4  Tracer + konverterer til Core ML ...")
example = torch.rand(1, 3, 640, 640)
traced = torch.jit.trace(tm, example)

import coremltools as ct
mlm = ct.convert(
    traced,
    inputs=[ct.ImageType(name="images", shape=(1, 3, 640, 640),
                         scale=1 / 255.0, color_layout=ct.colorlayout.RGB)],
    outputs=[ct.TensorType(name="output0")],
    minimum_deployment_target=ct.target.iOS16,
    compute_precision=ct.precision.FLOAT16,
)
mlm.save(OUT)
print(f"     Gemt: {OUT}")

print("3/4  Validerer mod onnxruntime ...")
import onnxruntime as ort
from PIL import Image

sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
rng = np.random.default_rng(7)
worst = 0.0
for _ in range(3):
    arr = rng.integers(0, 255, (640, 640, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    x = arr.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    ref = sess.run(None, {"images": x})[0]            # (1,7,8400)
    got = mlm.predict({"images": img})["output0"]
    # Sammenlign dér hvor det betyder noget: konfidenser + argmax-position
    ref_conf = ref[0, 4:7].max(0); got_conf = np.asarray(got)[0, 4:7].max(0)
    d = float(np.abs(ref_conf - got_conf).max())
    worst = max(worst, d)
    same_top = int(ref_conf.argmax()) == int(np.asarray(got_conf).argmax())
    print(f"     maks conf-afvigelse: {d:.4f}   samme top-detektion: {same_top}")

print("4/4  Resultat:", "✅ OK (fp16-afvigelse er normal < ~0.02)" if worst < 0.05
      else "⚠️ STOR afvigelse — send outputtet til Claude")
print("Færdig. Drag SwingClub.mlpackage ind i Xcode (kopiér, target ✓) og byg.")

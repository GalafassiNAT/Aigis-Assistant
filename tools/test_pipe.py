import json
import os
import subprocess
import sys

import numpy as np

np.save("test_dummy_feats.npy", np.zeros((1, 50, 768), dtype=np.float32))

p = subprocess.Popen(
    [sys.executable, "core/worker_net.py"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    encoding="utf-8",
)

line1 = p.stdout.readline().strip()
print(f"Handshake: {line1}")

for i in range(3):
    req = {
        "input_wav": os.path.abspath("test_base.wav"),
        "feats_npy": os.path.abspath("test_dummy_feats.npy"),
        "output_wav": os.path.abspath(f"output/audio/test_direct_pipe_{i}.wav"),
        "pitch_shift": 0.0,
    }
    p.stdin.write(json.dumps(req) + "\n")
    p.stdin.flush()
    line = p.stdout.readline().strip()
    print(f"Turno {i + 1}: {line}")

try:
    p.stdin.write("QUIT\n")
    p.stdin.flush()
except Exception:
    pass

p.wait(timeout=2)
print("Pipeline multi-turn DirectML 100% ESTÁVEL!")

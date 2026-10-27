import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import json

import numpy as np
import soundfile as sf
import torch
from scipy import signal
from transformers import HubertModel


def fast_load_16k(wav_path: str) -> np.ndarray:
    data, fs = sf.read(wav_path)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if fs != 16000:
        # Resample ultrarrápido usando resample_poly (ex: 24k -> 16k é 2/3)
        gcd = np.gcd(int(fs), 16000)
        up = 16000 // gcd
        down = int(fs) // gcd
        data = signal.resample_poly(data, up, down)
    return data.astype(np.float32)


hubert = HubertModel.from_pretrained("r3gm/hubert_base")
hubert.eval()

print("READY")
sys.stdout.flush()

for line in sys.stdin:
    line = line.strip()
    if not line or line == "QUIT":
        break
    try:
        req = json.loads(line)
        audio = fast_load_16k(req["input_wav"])
        inputs = torch.from_numpy(audio).float().unsqueeze(0)
        with torch.no_grad():
            outputs = hubert(inputs, output_hidden_states=True)
            feats = outputs.hidden_states[12]
        np.save(req["output_npy"], feats.cpu().numpy())
        print(json.dumps({"status": "ok"}))
        sys.stdout.flush()
    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}))
        sys.stdout.flush()

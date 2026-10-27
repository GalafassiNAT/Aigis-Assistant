import os
import sys
import traceback

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
from infer_rvc_python.lib.rmvpe import RMVPE
from scipy import signal


def fast_load_16k(wav_path: str) -> np.ndarray:
    data, fs = sf.read(wav_path)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if fs != 16000:
        gcd = np.gcd(int(fs), 16000)
        up = 16000 // gcd
        down = int(fs) // gcd
        data = signal.resample_poly(data, up, down)
    return data.astype(np.float32)


try:
    rmvpe = RMVPE("models/rvc/rmvpe.pt", is_half=False, device="cpu")
    onnx_path = "models/rvc/aigis/aigis_model.onnx"
    opts = ort.SessionOptions()
    opts.log_severity_level = 3

    # DirectML na Radeon RX 7600
    sess = ort.InferenceSession(
        onnx_path,
        sess_options=opts,
        providers=["DmlExecutionProvider", "CPUExecutionProvider"],
    )
    tgt_sr = 40000

    # Warmup
    _ = sess.run(
        ["audio"],
        {
            "feats": np.zeros((1, 50, 768), dtype=np.float32),
            "p_len": np.array([50], dtype=np.int64),
            "pitch": np.zeros((1, 50), dtype=np.int64),
            "pitchf": np.zeros((1, 50), dtype=np.float32),
            "sid": np.array([0], dtype=np.int64),
        },
    )
    print("READY")
    sys.stdout.flush()
except Exception as e:
    sys.stderr.write(f"INIT ERROR: {e}\n{traceback.format_exc()}\n")
    sys.stderr.flush()
    sys.exit(1)

for line in sys.stdin:
    line = line.strip()
    if not line or line == "QUIT":
        break
    try:
        req = json.loads(line)
        audio = fast_load_16k(req["input_wav"])
        f0 = rmvpe.infer_from_audio(audio, 0.03)

        pitch_shift = req.get("pitch_shift", 0.0)
        if pitch_shift != 0.0:
            f0 = f0 * (2.0 ** (pitch_shift / 12.0))

        feats_raw = np.load(req["feats_npy"])
        # Interpolação temporal 2x usando PyTorch em CPU rápido
        feats_t = torch.from_numpy(feats_raw).permute(0, 2, 1)
        feats_t = torch.nn.functional.interpolate(feats_t, scale_factor=2).permute(0, 2, 1)
        feats = feats_t.numpy()

        f0_mel_min = 1127 * np.log(1 + 50 / 700)
        f0_mel_max = 1127 * np.log(1 + 1100 / 700)
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = np.rint(f0_mel).astype(np.int64)

        p_len = min(feats.shape[1], len(f0))
        feats_ready = np.ascontiguousarray(feats[:, :p_len, :], dtype=np.float32)
        pitch_ready = np.ascontiguousarray(f0_coarse[:p_len].reshape(1, -1), dtype=np.int64)
        pitchf_ready = np.ascontiguousarray(f0[:p_len].reshape(1, -1), dtype=np.float32)
        sid_ready = np.array([0], dtype=np.int64)

        # Chunking adaptativo para estabilidade no DirectML (limite seguro: 1500 frames / 15s)
        max_chunk = 1500
        overlap = 50
        hop = 400

        if p_len <= max_chunk:
            out = sess.run(
                ["audio"],
                {
                    "feats": feats_ready,
                    "p_len": np.array([p_len], dtype=np.int64),
                    "pitch": pitch_ready,
                    "pitchf": pitchf_ready,
                    "sid": sid_ready,
                },
            )
            audio_out = out[0].squeeze()
        else:
            # Overlap-add com crossfade linear
            overlap_samples = overlap * hop
            step = max_chunk - overlap
            total_audio_samples = p_len * hop
            audio_out = np.zeros(total_audio_samples, dtype=np.float32)
            weight_mask = np.zeros(total_audio_samples, dtype=np.float32)

            fade_in = np.linspace(0.0, 1.0, overlap_samples, dtype=np.float32)
            fade_out = np.linspace(1.0, 0.0, overlap_samples, dtype=np.float32)

            start = 0
            while start < p_len:
                end = min(start + max_chunk, p_len)
                chunk_len = end - start

                out = sess.run(
                    ["audio"],
                    {
                        "feats": feats_ready[:, start:end, :],
                        "p_len": np.array([chunk_len], dtype=np.int64),
                        "pitch": pitch_ready[:, start:end],
                        "pitchf": pitchf_ready[:, start:end],
                        "sid": sid_ready,
                    },
                )
                chunk_audio = out[0].squeeze()
                c_samples = len(chunk_audio)

                window = np.ones(c_samples, dtype=np.float32)
                if start > 0:
                    window[:overlap_samples] = fade_in
                if end < p_len:
                    window[-overlap_samples:] = fade_out

                a_start = start * hop
                a_end = a_start + c_samples
                audio_out[a_start:a_end] += chunk_audio * window
                weight_mask[a_start:a_end] += window

                if end == p_len:
                    break
                start += step

            nonzero = weight_mask > 0
            audio_out[nonzero] /= weight_mask[nonzero]

        max_val = np.max(np.abs(audio_out))
        if max_val > 0.95:
            audio_out = audio_out * (0.95 / max_val)

        sf.write(req["output_wav"], audio_out, tgt_sr)
        print(json.dumps({"status": "ok"}))
        sys.stdout.flush()
    except Exception as e:
        err_msg = f"{e}\n{traceback.format_exc()}"
        sys.stderr.write(f"INFER ERROR: {err_msg}\n")
        sys.stderr.flush()
        print(json.dumps({"status": "error", "message": str(e)}))
        sys.stdout.flush()

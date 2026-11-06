import gc
import json
import os
import sys
import traceback

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
    import faiss

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rmvpe = RMVPE("models/rvc/rmvpe.pt", is_half=False, device=device)
    onnx_path = "models/rvc/aigis/aigis_model.onnx"
    opts = ort.SessionOptions()
    opts.log_severity_level = 3

    # Dicionário acústico anatômico FAISS para restaurar timbre Persona 3 autêntico
    index_path = "models/rvc/aigis/AigisMrM0dZ.index"
    index = None
    big_npy = None
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
        big_npy = index.reconstruct_n(0, index.ntotal)

    # Detecta automaticamente provedores compativeis com Linux ou Windows
    available_providers = set(ort.get_available_providers())
    preferred_order = [
        "ROCMExecutionProvider",
        "CUDAExecutionProvider",
        "DmlExecutionProvider",
        "CPUExecutionProvider",
    ]
    providers = [p for p in preferred_order if p in available_providers]
    if not providers:
        providers = ["CPUExecutionProvider"]

    sess = ort.InferenceSession(
        onnx_path,
        sess_options=opts,
        providers=providers,
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
    if line == "QUIT":
        break
    if not line:
        continue
    try:
        req = json.loads(line)
        audio = fast_load_16k(req["input_wav"])
        f0 = rmvpe.infer_from_audio(audio, 0.03)

        pitch_shift = req.get("pitch_shift", 0.0)
        if pitch_shift != 0.0:
            f0 = f0 * (2.0 ** (pitch_shift / 12.0))

        # 1. Suavização natural de pitch (elimina tremolo/flutter sem saltos de oitava)
        voiced = f0 > 0
        if np.any(voiced):
            f0[voiced] = signal.medfilt(f0[voiced], kernel_size=3)
        # Zera ruídos espúrios isolados abaixo do piso vocal da Aigis (< 110Hz)
        f0[(f0 > 0) & (f0 < 110.0)] = 0.0

        feats_raw = np.load(req["feats_npy"])
        feats0 = torch.from_numpy(feats_raw)

        # 2. Projeção de Faringe via FAISS k-NN (Dicionário de Timbre Aigis)
        index_rate = 0.60
        protect = 0.33
        if index is not None and index_rate > 0:
            npy = feats_raw[0].astype(np.float32)
            score, ix = index.search(npy, k=8)
            weight = np.square(1.0 / np.maximum(score, 1e-6))
            weight /= weight.sum(axis=1, keepdims=True)
            npy_retr = np.sum(big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)
            feats_proj = torch.from_numpy(npy_retr).unsqueeze(0) * index_rate + (1.0 - index_rate) * feats0
        else:
            feats_proj = feats0

        # 3. Interpolação LINEAR contínua (elimina 100% o artefato metálico de 50Hz)
        feats_t = torch.nn.functional.interpolate(
            feats_proj.permute(0, 2, 1), scale_factor=2, mode="linear", align_corners=False
        ).permute(0, 2, 1)
        feats0_t = torch.nn.functional.interpolate(
            feats0.permute(0, 2, 1), scale_factor=2, mode="linear", align_corners=False
        ).permute(0, 2, 1)

        # 4. Proteção de Consoantes Surdas e Plosivas (impede aspereza metálica em 't', 'k', 's')
        eff_len = min(feats_t.shape[1], len(f0))
        pitchf_sub = torch.from_numpy(f0[:eff_len].astype(np.float32)).unsqueeze(0)
        pitch_mask = pitchf_sub.clone()
        pitch_mask[pitchf_sub > 0] = 1.0
        pitch_mask[pitchf_sub < 1] = protect
        pitch_mask = pitch_mask.unsqueeze(-1)
        feats = (feats_t[:, :eff_len, :] * pitch_mask + feats0_t[:, :eff_len, :] * (1.0 - pitch_mask)).numpy()

        f0_mel_min = 1127 * np.log(1 + 50 / 700)
        f0_mel_max = 1127 * np.log(1 + 1100 / 700)
        f0_mel = 1127 * np.log(1 + f0[:eff_len] / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = np.rint(f0_mel).astype(np.int64)

        p_len = eff_len
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
        del audio, f0, feats_raw, feats_t, feats, feats_ready, pitch_ready, pitchf_ready, audio_out
        gc.collect()
        print(json.dumps({"status": "ok"}))
        sys.stdout.flush()
    except Exception as e:
        err_msg = f"{e}\n{traceback.format_exc()}"
        sys.stderr.write(f"INFER ERROR: {err_msg}\n")
        sys.stderr.flush()
        print(json.dumps({"status": "error", "message": str(e)}))
        sys.stdout.flush()

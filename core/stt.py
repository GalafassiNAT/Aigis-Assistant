import collections
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

# Garante raiz no sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class SpeechRecognizer:
    """
    Motor local de Speech-To-Text usando faster-whisper (CTranslate2).
    Captura áudio do microfone padrão com detecção simples de silêncio (VAD de energia)
    e transcreve localmente com baixa latência em CPU.
    """

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        sample_rate: int = 16000,
        silence_threshold: float = 0.0020,
        silence_duration: float = 0.8,
        min_speech_duration: float = 0.5,
        input_device: str | int | None = None,
    ):
        import os

        self.sample_rate = sample_rate
        # Permite sobrescrever via ambiente (default ajustado para 0.0080 para evitar falso positivo em Linux)
        self.silence_threshold = float(os.getenv("AUDIO_SILENCE_THRESHOLD", 0.0080 if silence_threshold == 0.0020 else silence_threshold))
        self.silence_duration = float(os.getenv("AUDIO_SILENCE_TIMEOUT", silence_duration))
        self.min_speech_duration = min_speech_duration
        self.input_device = self._resolve_device(input_device or os.getenv("AUDIO_INPUT_DEVICE"))

        print(f"[STT] Inicializando modelo Whisper '{model_size}' ({device}/{compute_type})...")
        cpu_threads = int(os.getenv("WHISPER_CPU_THREADS", "8"))
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
        )
        print("[STT] Modelo Whisper pronto!")

        self.audio_queue = queue.Queue()
        self.is_recording = False
        self._stream = None
        self._listen_thread = None

    def _resolve_device(self, dev: str | int | None) -> int | str | None:
        """Resolve nome parcial ou índice de dispositivo de áudio."""
        if dev is None or dev == "":
            return None
        try:
            return int(dev)
        except ValueError:
            pass

        # Procura por substring no nome
        for idx, d in enumerate(sd.query_devices()):
            if d.get("max_input_channels", 0) > 0 and dev.lower() in d.get("name", "").lower():
                return idx
        return None

    def _audio_callback(self, indata, frames, time_info, status):
        """Callback contínuo do sounddevice."""
        if status:
            pass
        self.audio_queue.put(indata.copy())

    def transcribe(self, audio_data: np.ndarray, language: str | None = "pt") -> str:
        """
        Transcreve um bloco numpy de float32 a 16kHz.
        Usa beam_size=1 (greedy) para máxima velocidade em CPU com prompt âncora.
        """
        segments, info = self.model.transcribe(
            audio_data,
            beam_size=1,
            language=language,
            initial_prompt="Conversa em português com a assistente pessoal Aigis.",
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=250),
        )
        texto = " ".join([seg.text.strip() for seg in segments]).strip()

        # Pós-processamento fonético para corrigir variações comuns de STT em PT-BR
        correcoes = [
            (r"\bno\s+[Ee]ntiais\b", "Aigis"),
            (r"\b[Ee]ntiais\b", "Aigis"),
            (r"\b[Dd]e certe\b", "disserte"),
            (r"\b[Dd]eu uma breve\b", "dê uma breve"),
            (r"\b[Ée] isso\b", "Aigis"),
            (r"\b[Aa]egis\b", "Aigis"),
            (r"\b[Aa]igys\b", "Aigis"),
            (r"\b[Áá]iguis\b", "Aigis"),
            (r"\b[Aa]iguis\b", "Aigis"),
            (r"\b[Aa]yges\b", "Aigis"),
            (r"\b[Hh]aigis\b", "Aigis"),
            (r"server\.descorde", "servidor do Discord"),
            (r"\bdescorde\b", "Discord"),
            (r"\bdescord\b", "Discord"),
        ]
        import re

        for pat, rep in correcoes:
            texto = re.sub(pat, rep, texto, flags=re.IGNORECASE)

        return texto

    def listen_and_transcribe_once(
        self, timeout_silence: float = 0.8, stop_event: threading.Event | None = None
    ) -> str:
        """
        Bloqueia até que o usuário fale algo no microfone e faça uma pausa de silêncio ágil (0.8s).
        Retorna o texto transcrito. Respeita stop_event para encerramento gracioso com Ctrl+C.
        """
        chunk_size = int(self.sample_rate * 0.1)  # blocos de 100ms
        audio_buffer = []
        is_speaking = False
        silence_start = None

        channels = 1
        if self.input_device is not None:
            try:
                dev_info = sd.query_devices(self.input_device)
                channels = min(1, dev_info.get("max_input_channels", 1))
            except Exception:
                channels = 1

        # Auto-calibracao rápida de ruido ambiente nos primeiros 3 blocos (300ms) se limiar nao foi fixado
        calibrated_threshold = self.silence_threshold
        with sd.InputStream(
            device=self.input_device,
            samplerate=self.sample_rate,
            channels=channels,
            dtype="float32",
            blocksize=chunk_size,
        ) as stream:
            # Mede 2 frames de ambiente para adaptar ao ganho do mic
            ambient = []
            for _ in range(2):
                d, _ = stream.read(chunk_size)
                ambient.append(np.sqrt(np.mean(d**2)))
            noise_floor = max(ambient) if ambient else 0.003
            calibrated_threshold = max(self.silence_threshold, noise_floor * 2.5)

            print(f"\n🎤 [Aigis está ouvindo... (ruído={noise_floor:.4f}, corte={calibrated_threshold:.4f})]")

            pre_roll = collections.deque(maxlen=3)
            while True:
                if stop_event and stop_event.is_set():
                    return ""

                data, _ = stream.read(chunk_size)
                if data.ndim > 1:
                    data = np.mean(data, axis=1, keepdims=True)
                rms = np.sqrt(np.mean(data**2))

                if rms > calibrated_threshold:
                    if not is_speaking:
                        print("🗣️ Fala detectada...")
                        is_speaking = True
                        audio_buffer.extend(list(pre_roll))
                    audio_buffer.append(data)
                    silence_start = None
                elif is_speaking:
                    audio_buffer.append(data)
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start >= timeout_silence:
                        # Usuário terminou de falar
                        break
                else:
                    pre_roll.append(data)

        if not audio_buffer:
            return ""

        audio_full = np.concatenate(audio_buffer, axis=0).flatten()
        duration = len(audio_full) / self.sample_rate
        if duration < self.min_speech_duration:
            return ""

        print(f"⚡ Transcrevendo ({duration:.1f}s de áudio)...")
        t0 = time.time()
        texto = self.transcribe(audio_full)
        lat = (time.time() - t0) * 1000
        print(f'📝 Transcrição ({lat:.0f}ms): "{texto}"')
        return texto

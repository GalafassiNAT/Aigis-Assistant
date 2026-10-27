"""
Módulo de Síntese e Reprodução de Voz para o Projeto Aigis.
Arquitetura Dual-Engine Residente:
- Inglês: Motor Kokoro ('af_heart') + Aigis RVC v2 (timbre 1:1, fluidez de jogo).
- Português: Motor Edge-TTS ('pt-BR-FranciscaNeural') com a calibração pura do áudio de referência (test_aigis_intro.wav).
- Normalização fonética automática:
    - Inglês: 'Eyegis' -> 'Ah-i-gis' / 'Eye-gis'
    - Português: 'Áiguis' -> 'Ah-i-guis' com Hard G ('gu' como em Guilherme)
- Daemons ultrarrápidos residentes em RAM mantêm a latência de síntese abaixo de 3 segundos.
- Saída de áudio centralizada em 'output/audio/'.
"""

import asyncio
import json
import os
import re
import subprocess
import sys
import time

import edge_tts
import sounddevice as sd
import soundfile as sf
from kokoro_onnx import Kokoro
from loguru import logger


class VoiceSynthesizer:
    def __init__(self, voice_pt: str = "pt-BR-FranciscaNeural", kokoro_voice_en: str = "af_heart"):
        self.voice_pt = voice_pt
        self.kokoro_voice_en = kokoro_voice_en
        self.core_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.abspath(os.path.join(self.core_dir, ".."))
        self.output_dir = os.path.join(self.project_root, "output", "audio")
        os.makedirs(self.output_dir, exist_ok=True)

        logger.info("[VoiceSynthesizer] Inicializando daemons vocais da Aigis em memória...")
        self._start_hubert()
        self._start_net()

        # 3. Kokoro ONNX para Inglês nativo expressivo
        kokoro_model = os.path.join(self.project_root, "models", "kokoro", "kokoro-v1.0.onnx")
        kokoro_voices = os.path.join(self.project_root, "models", "kokoro", "voices-v1.0.bin")
        if os.path.exists(kokoro_model) and os.path.exists(kokoro_voices):
            self.kokoro = Kokoro(kokoro_model, kokoro_voices)
        else:
            self.kokoro = None

    def _start_hubert(self):
        hubert_script = os.path.join(self.core_dir, "worker_hubert.py")
        self.proc_hubert = subprocess.Popen(
            [sys.executable, "-u", hubert_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        while True:
            line = self.proc_hubert.stdout.readline()
            if "READY" in line:
                break
            if self.proc_hubert.poll() is not None:
                err = self.proc_hubert.stderr.read()
                raise RuntimeError(f"Falha ao iniciar worker HuBERT: {err}")

    def _start_net(self):
        net_script = os.path.join(self.core_dir, "worker_net.py")
        self.proc_net = subprocess.Popen(
            [sys.executable, "-u", net_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        while True:
            line = self.proc_net.stdout.readline()
            if "READY" in line:
                break
            if self.proc_net.poll() is not None:
                err = self.proc_net.stderr.read()
                raise RuntimeError(f"Falha ao iniciar worker Aigis Net: {err}")

        logger.info(
            "[VoiceSynthesizer] Todos os motores neurais prontos em RAM (Síntese ultrarrápida ativa)."
        )

    def _detect_lang(self, text: str) -> str:
        """Heurística rápida de detecção pt vs en sem dependências externas."""
        en_markers = {
            "the",
            "is",
            "are",
            "you",
            "i",
            "my",
            "your",
            "hello",
            "hi",
            "how",
            "what",
            "trying",
            "english",
            "voice",
            "now",
            "am",
            "fact",
            "future",
            "guarantee",
            "master",
            "ready",
        }
        words = {w.strip(".,!?\"'").lower() for w in text.split()}
        matches = len(words.intersection(en_markers))
        return "en" if matches >= 2 or (len(words) <= 3 and matches >= 1) else "pt"

    def _normalize_phonetics(self, text: str, lang: str) -> str:
        """Garante a pronúncia canônica autêntica de 'Aigis' (Ah-i-gis com Hard G)."""
        if lang == "en":
            # No Kokoro, 'Eyegis' reproduz perfeitamente o 'Ah-i-gis' / 'Eye-gis'
            return re.sub(r"\bAigis\b", "Eyegis", text, flags=re.IGNORECASE)
        else:
            # No português brasileiro (Edge-TTS), 'Áiguis' gera o Hard G autêntico ('gu' como em Guilherme)
            return re.sub(r"\bAigis\b", "Áiguis", text, flags=re.IGNORECASE)

    async def _generate_base_tts(self, text: str, output_pcm_path: str) -> tuple[str, float]:
        lang = self._detect_lang(text)
        clean_text = self._normalize_phonetics(text, lang)

        if lang == "en" and self.kokoro is not None:
            # Kokoro Heart entrega entonação expressiva natural 1:1
            samples, sr = self.kokoro.create(
                clean_text, voice=self.kokoro_voice_en, speed=0.90, lang="en-us"
            )
            sf.write(output_pcm_path, samples, sr, format="WAV", subtype="PCM_16")
            pitch_shift = 0.0
        else:
            # Francisca idêntica à de test_aigis_intro.wav (ritmo e tom limpos e contínuos)
            temp_raw = output_pcm_path + ".raw.mp3"
            communicate = edge_tts.Communicate(clean_text, self.voice_pt, rate="-4%", pitch="-1Hz")
            await communicate.save(temp_raw)
            try:
                data, fs = sf.read(temp_raw)
                sf.write(output_pcm_path, data, fs, format="WAV", subtype="PCM_16")
            finally:
                if os.path.exists(temp_raw):
                    try:
                        os.remove(temp_raw)
                    except OSError:
                        pass
            pitch_shift = 0.0

        return lang, pitch_shift

    def convert_voice(
        self, input_wav_path: str, output_wav_path: str, pitch_shift: float = 0.0
    ) -> str:
        timestamp = int(time.time() * 1000)
        temp_feats = os.path.join(self.output_dir, f"temp_feats_{timestamp}.npy")

        # Recuperação resiliente automática caso algum daemon tenha sido encerrado pelo SO
        if self.proc_hubert.poll() is not None:
            logger.warning("[VoiceSynthesizer] Reiniciando HuBERT worker...")
            self._start_hubert()
        if self.proc_net.poll() is not None:
            logger.warning("[VoiceSynthesizer] Reiniciando Aigis Net worker...")
            self._start_net()

        try:
            # 1. HuBERT Feature Extraction
            req1 = {
                "input_wav": os.path.abspath(input_wav_path),
                "output_npy": os.path.abspath(temp_feats),
            }
            self.proc_hubert.stdin.write(json.dumps(req1) + "\n")
            self.proc_hubert.stdin.flush()
            line1 = self.proc_hubert.stdout.readline()
            if not line1:
                raise RuntimeError(f"HuBERT worker caiu: {self.proc_hubert.stderr.read()}")
            r1 = json.loads(line1)
            if r1.get("status") != "ok":
                raise RuntimeError(f"Erro no estágio HuBERT: {r1.get('message')}")

            # 2. Pitch Tracking + Aigis Net Synthesis acelerado na GPU DirectML
            req2 = {
                "input_wav": os.path.abspath(input_wav_path),
                "feats_npy": os.path.abspath(temp_feats),
                "output_wav": os.path.abspath(output_wav_path),
                "pitch_shift": pitch_shift,
            }
            self.proc_net.stdin.write(json.dumps(req2) + "\n")
            self.proc_net.stdin.flush()
            line2 = self.proc_net.stdout.readline()
            if not line2:
                err_net = self.proc_net.stderr.read()
                raise RuntimeError(f"Aigis Net worker caiu (rc={self.proc_net.poll()}): {err_net}")
            r2 = json.loads(line2)
            if r2.get("status") != "ok":
                raise RuntimeError(f"Erro na síntese da Aigis: {r2.get('message')}")

            return output_wav_path
        finally:
            if os.path.exists(temp_feats):
                try:
                    os.remove(temp_feats)
                except OSError:
                    pass

    async def speak(
        self,
        text: str,
        output_filename: str = "aigis_live.wav",
        play_audio: bool = True,
    ) -> str:
        """Sintetiza texto com a voz da Aigis e salva na pasta output/audio/."""
        timestamp = int(time.time() * 1000)
        temp_base = os.path.join(self.output_dir, f"temp_base_{timestamp}.wav")
        output_path = os.path.join(self.output_dir, output_filename)

        try:
            lang, pitch_shift = await self._generate_base_tts(text, temp_base)
            self.convert_voice(temp_base, output_path, pitch_shift=pitch_shift)

            if play_audio:
                self.play(output_path)

            return output_path
        finally:
            if os.path.exists(temp_base):
                try:
                    os.remove(temp_base)
                except OSError:
                    pass

    @staticmethod
    def play(wav_path: str, block: bool = True):
        data, fs = sf.read(wav_path)
        sd.play(data, fs)
        if block:
            sd.wait()

    def close(self):
        """Encerra os daemons graciosamente."""
        for p in [self.proc_hubert, self.proc_net]:
            if p and p.poll() is None:
                try:
                    p.stdin.write("QUIT\n")
                    p.stdin.flush()
                    p.wait(timeout=2)
                except Exception:
                    p.kill()

    def __del__(self):
        self.close()


if __name__ == "__main__":
    synthesizer = VoiceSynthesizer()
    try:
        asyncio.run(
            synthesizer.speak(
                "Sou Aigis, sistemas vocais totalmente operacionais.",
                "teste_rapido.wav",
                play_audio=True,
            )
        )
    finally:
        synthesizer.close()

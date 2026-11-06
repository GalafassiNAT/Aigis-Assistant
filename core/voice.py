"""
Módulo de Síntese e Reprodução de Voz em Tempo Real para o Projeto Aigis (100% GPU).
Arquitetura:
- Português (PT-BR): F5-TTS PT-BR ('viperdf/f5-tts_pt-br' + Vocos 24kHz em GPU FP16) +
  Trato Vocal In-VRAM RVC v2 40kHz ('AigisMrM0dZ_300e_5700s.pth' PyTorch GPU).
- Inglês (EN): Motor Kokoro ONNX ('af_heart') + Trato Vocal Aigis.
- Normalização fonética e numérica completa:
  - 'Aigis' -> 'áiguis' (Hard G autêntico)
  - Expansão automática de numerais/datas ('5' -> 'cinco', '2026' -> 'dois mil e vinte e seis')
  - Higienização de travessões, markdown e siglas para evitar palavras engolidas
- Reprodução serializada com Lock assíncrono, rampa de 8ms e tail-pad de 60ms para PipeWire/Linux.
"""

import asyncio
import glob
import io
import os
import re
import sys
import threading
import time
import wave

if sys.platform.startswith("linux"):
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
    os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
    os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

import contextlib
import numpy as np

np.float = float
np.complex = complex
import sounddevice as sd
import soundfile as sf
import torch
from kokoro_onnx import Kokoro
from loguru import logger

from core.vocal_tract import InVRAMVocalTractAigis

_UNIDADES = (
    "zero",
    "um",
    "dois",
    "três",
    "quatro",
    "cinco",
    "seis",
    "sete",
    "oito",
    "nove",
    "dez",
    "onze",
    "doze",
    "treze",
    "quatorze",
    "quinze",
    "dezesseis",
    "dezessete",
    "dezoito",
    "dezenove",
)

_DEZENAS = (
    "",
    "",
    "vinte",
    "trinta",
    "quarenta",
    "cinquenta",
    "sessenta",
    "setenta",
    "oitenta",
    "noventa",
)

_CENTENAS = (
    "",
    "cento",
    "duzentos",
    "trezentos",
    "quatrocentos",
    "quinhentos",
    "seiscentos",
    "setecentos",
    "oitocentos",
    "novecentos",
)


def _int_to_pt_br(n: int) -> str:
    """Converte inteiros de 0 a 999.999 para extenso em português brasileiro sem dependências externas."""
    if n < 0:
        return "menos " + _int_to_pt_br(-n)
    if n < 20:
        return _UNIDADES[n]
    if n < 100:
        d, u = divmod(n, 10)
        return _DEZENAS[d] if u == 0 else f"{_DEZENAS[d]} e {_UNIDADES[u]}"
    if n == 100:
        return "cem"
    if n < 1000:
        c, r = divmod(n, 100)
        return _CENTENAS[c] if r == 0 else f"{_CENTENAS[c]} e {_int_to_pt_br(r)}"
    if n < 1_000_000:
        m, r = divmod(n, 1000)
        prefix = "mil" if m == 1 else f"{_int_to_pt_br(m)} mil"
        if r == 0:
            return prefix
        if r < 100 or r % 100 == 0:
            return f"{prefix} e {_int_to_pt_br(r)}"
        return f"{prefix} {_int_to_pt_br(r)}"
    return " ".join(_UNIDADES[int(d)] for d in str(n))


def expand_numbers_pt_br(text: str) -> str:
    """Expande horários, porcentagens e numerais cardinais para extenso em PT-BR."""
    # Horários ex: 14:30 -> quatorze e trinta
    def _repl_time(m: re.Match) -> str:
        h, mins = int(m.group(1)), int(m.group(2))
        if mins == 0:
            return f"{_int_to_pt_br(h)} horas"
        return f"{_int_to_pt_br(h)} e {_int_to_pt_br(mins)}"

    text = re.sub(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", _repl_time, text)

    # Porcentagens ex: 50% -> cinquenta por cento
    text = re.sub(
        r"\b(\d+)\s*%",
        lambda m: f"{_int_to_pt_br(int(m.group(1)))} por cento",
        text,
    )

    # Decimais simples ex: 3,5 ou 3.5 -> três vírgula cinco
    text = re.sub(
        r"\b(\d+)[,.](\d+)\b",
        lambda m: f"{_int_to_pt_br(int(m.group(1)))} vírgula {_int_to_pt_br(int(m.group(2)))}",
        text,
    )

    # Inteiros gerais ex: 5 -> cinco, 2026 -> dois mil e vinte e seis
    text = re.sub(
        r"\b(\d{1,6})\b",
        lambda m: _int_to_pt_br(int(m.group(1))),
        text,
    )
    return text


_ANGLICISMS_PTBR: dict[str, str] = {
    # Linguagens e Runtimes
    "TypeScript": "Táipscripte",
    "JavaScript": "Javascripte",
    "Python": "Páiton",
    "Pytest": "Páiteste",
    "Node.js": "Nôude Jota-Esse",
    "NodeJS": "Nôude Jota-Esse",
    "Node": "Nôude",
    "Next.js": "Nékst Jota-Esse",
    "NextJS": "Nékst Jota-Esse",
    "Next": "Nékst",
    "React": "Ríéct",
    "Rust": "Râst",
    "Golang": "Goulêngue",
    "Django": "Djângo",
    "FastAPI": "Fást-a-pê-í",
    "Flutter": "Fláter",
    "Kotlin": "Kótlin",
    "HTML": "Agá-tê-eme-éle",
    "CSS": "Cê-esse-esse",
    "SQL": "Esse-quê-ele",
    "JSON": "Jêison",
    "YAML": "Iâmel",
    "XML": "Xis-eme-éle",
    "Bash": "Bésh",
    "PowerShell": "Póuer-shél",
    "Shell": "Shél",
    # Infraestrutura e Ferramentas
    "Docker": "Dóquer",
    "Kubernetes": "Cubernetes",
    "Desktop": "Désktop",
    "Linux": "Línux",
    "Windows": "Uíndous",
    "Discord": "Díscorde",
    "Obsidian": "Obsídian",
    "PostgreSQL": "Póstgres",
    "Postgres": "Póstgres",
    "SQLite": "Esse-quê-ele-láite",
    "MySQL": "Mái-esse-quê-ele",
    "Redis": "Rédis",
    "GitHub": "Guít-râb",
    "GitLab": "Guít-léb",
    "Git": "Guít",
    "Vulkan": "Vúlcan",
    "Whisper": "Uísper",
    "DeepSeek": "Díp-síik",
    "OpenAI": "Ôupen-êi-ái",
    "Llama": "Lhama",
    "GGUF": "Gê-gê-u-éfe",
    "CUDA": "Cúda",
    "ROCm": "Róc-eme",
    "ONNX": "Ônix",
    # Hardware e Componentes
    "CPU": "Cê-pê-ú",
    "CPUs": "Cê-pê-ús",
    "GPU": "Gê-pê-ú",
    "GPUs": "Gê-pê-ús",
    "VRAM": "Vê-râm",
    "RAM": "Râm",
    "SSD": "Esse-esse-dê",
    "SSDs": "Esse-esse-dês",
    "HDD": "Agá-dê",
    "HDDs": "Agá-dês",
    "USB": "U-esse-bê",
    # Siglas Técnicas
    "API": "Á-Pê-Í",
    "APIs": "Á-Pê-Ís",
    "CLI": "Cê-éle-í",
    "GUI": "Gúi",
    "SDK": "Esse-dê-cá",
    "LLM": "Éle-éle-ême",
    "LLMs": "Éle-éle-êmes",
    "STT": "Esse-tê-tê",
    "TTS": "Tê-tê-esse",
    "RVC": "Érre-vê-cê",
    "AI": "A-Í",
    # Engenharia de Software
    "Backend": "Béqui-ênd",
    "Frontend": "Frônt-ênd",
    "Fullstack": "Ful-stéqui",
    "Frameworks": "Freim-uôrques",
    "Framework": "Freim-uôrque",
    "Libraries": "Láibraris",
    "Library": "Láibrari",
    "Libs": "Líbis",
    "Lib": "Líbi",
    "Deploys": "Deplóis",
    "Deploy": "Deplói",
    "Bugs": "Bágues",
    "Bug": "Bágue",
    "Debugging": "Dibáguing",
    "Debug": "Dibágue",
    "Features": "Fítchurs",
    "Feature": "Fítchur",
    "Releases": "Relíises",
    "Release": "Relíise",
    "Pull Requests": "Pul Riquésts",
    "Pull Request": "Pul Riquést",
    "PR": "Pê-Érre",
    "Commits": "Comítis",
    "Commit": "Comíti",
    "Merge": "Mêrge",
    "Branches": "Brântches",
    "Branch": "Brântch",
    "Repos": "Rêpos",
    "Repo": "Rêpo",
    "Threads": "Tréds",
    "Thread": "Tréd",
    "Loops": "Lúps",
    "Loop": "Lúp",
    "Streaming": "Stríming",
    "Stream": "Strím",
    "Hardware": "Rárduér",
    "Software": "Sóftuér",
    "Firmware": "Fírmuér",
    "Online": "On-láine",
    "Offline": "Of-láine",
    "Prompts": "Prômpts",
    "Prompt": "Prômpt",
    "Reviews": "Reviús",
    "Review": "Reviú",
    "Links": "Línques",
    "Link": "Línque",
    "Setup": "Setápe",
    "Saves": "Sêives",
    "Save": "Sêive",
    "Backups": "Béqui-ápes",
    "Backup": "Béqui-ápe",
    "Layouts": "Lêi-áutes",
    "Layout": "Lêi-áute",
    "Feedbacks": "Fídi-béquis",
    "Feedback": "Fídi-béqui",
    "Tokens": "Tôquens",
    "Token": "Tôquen",
    "Benchmarks": "Bêntch-marks",
    "Benchmark": "Bêntch-mark",
    "Plugins": "Plúguins",
    "Plugin": "Plúguin",
    "Builds": "Bílds",
    "Build": "Bíld",
    "Endpoints": "Énd-points",
    "Endpoint": "Énd-point",
    "Payloads": "Pêi-louds",
    "Payload": "Pêi-loud",
    "Callbacks": "Cól-béquis",
    "Callback": "Cól-béqui",
    "Middlewares": "Mídel-uérs",
    "Middleware": "Mídel-uér",
    "Websites": "Uéb-sáites",
    "Website": "Uéb-sáite",
    "Web": "Uéb",
    "Status": "Státus",
    "Default": "Defôult",
    "Logs": "Lógues",
    "Log": "Lógue",
    "Check": "Tchéque",
    "Upload": "Up-lôud",
    "Download": "Doun-lôud",
    "Devs": "Dévis",
    "Dev": "Déve",
    "Apps": "Épis",
    "App": "Épe",
    "Packages": "Péquedjes",
    "Package": "Péquedj",
    "Scripts": "Escrípts",
    "Script": "Escrípt",
    "Servers": "Sérvers",
    "Server": "Sérver",
    "Localhost": "Loucal-rôust",
    "Databases": "Dêita-bêises",
    "Database": "Dêita-bêise",
    "Tasks": "Tásks",
    "Task": "Tásk",
    "Background": "Béqui-gráund",
    "Realtime": "Ríal-táime",
    "Real-time": "Ríal-táime",
}

_ANGLICISMS_KEYS = sorted(_ANGLICISMS_PTBR.keys(), key=len, reverse=True)
_ANGLICISMS_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _ANGLICISMS_KEYS) + r")\b",
    re.IGNORECASE,
)
_ANGLICISMS_LOOKUP = {k.lower(): v for k, v in _ANGLICISMS_PTBR.items()}


class VoiceSynthesizer:
    def __init__(
        self,
        voice_pt: str = "pt-BR-FranciscaNeural",
        kokoro_voice_en: str = "af_heart",
    ):
        self.voice_pt = voice_pt
        self.kokoro_voice_en = kokoro_voice_en
        self.core_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.abspath(os.path.join(self.core_dir, ".."))
        self.output_dir = os.path.join(self.project_root, "output", "audio")
        os.makedirs(self.output_dir, exist_ok=True)

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._play_lock = asyncio.Lock()
        self._gpu_lock = threading.Lock()

        logger.info(f"[VoiceSynthesizer] Inicializando motores neurais da Aigis em {self.device}...")

        # 1. Trato Vocal Residente 100% em GPU (HuBERT + RMVPE + FAISS + NSF-HiFiGAN PyTorch 40kHz)
        self.vocal_tract = InVRAMVocalTractAigis()

        # 2. Motor Prosódico Principal PT-BR em GPU: F5-TTS PT-BR (viperdf + Vocos 24kHz em FP16)
        self.f5_model = None
        self.f5_vocoder = None
        self.f5_ref_audio = None
        self.f5_ref_text = None
        self._init_f5_ptbr()

        # 3. Fallback Local Leve PT-BR (Piper Dii)
        dii_model = os.path.join(self.project_root, "models", "piper", "voices", "pt_BR-dii-high.onnx")
        dii_json = os.path.join(self.project_root, "models", "piper", "voices", "pt_BR-dii-high.onnx.json")
        self.pt_base_piper = None
        self.syn_cfg = None
        if os.path.exists(dii_model) and os.path.exists(dii_json):
            try:
                import piper

                self.pt_base_piper = piper.PiperVoice.load(dii_model, config_path=dii_json)
                self.syn_cfg = piper.SynthesisConfig(
                    length_scale=1.05,
                    noise_scale=0.333,
                    noise_w_scale=0.333,
                )
            except Exception as e:
                logger.debug(f"[VoiceSynthesizer] Fallback Piper indisponível: {e}")

        # 4. Motor Kokoro ONNX para Inglês
        kokoro_model = os.path.join(self.project_root, "models", "kokoro", "kokoro-v1.0.onnx")
        kokoro_voices = os.path.join(self.project_root, "models", "kokoro", "voices-v1.0.bin")
        self.kokoro = None
        if os.path.exists(kokoro_model) and os.path.exists(kokoro_voices):
            try:
                self.kokoro = Kokoro(kokoro_model, kokoro_voices)
                logger.info("[VoiceSynthesizer] Motor Kokoro ONNX carregado para inglês.")
            except Exception as e:
                logger.warning(f"[VoiceSynthesizer] Falha ao carregar Kokoro: {e}")

        # 5. Piper ONNX Destilado Opcional via .env (para quando o treino Colab concluir)
        self.use_piper = os.getenv("USE_PIPER_TTS", "false").lower() == "true"
        piper_model = os.path.join(self.project_root, "models", "piper", "aigis_tts_ptbr.onnx")
        piper_json = os.path.join(self.project_root, "models", "piper", "aigis_tts_ptbr.onnx.json")
        self.piper_voice = None
        if self.use_piper and os.path.exists(piper_model) and os.path.exists(piper_json):
            try:
                import piper

                self.piper_voice = piper.PiperVoice.load(piper_model, config_path=piper_json)
                logger.info("[VoiceSynthesizer] Motor Piper Destilado ativo via USE_PIPER_TTS=true.")
            except Exception as e:
                logger.warning(f"[VoiceSynthesizer] Falha ao carregar Piper Destilado: {e}")

        # 6. Cache de Immediate Audio Ack (<50ms)
        self._ack_cache: dict[str, tuple[np.ndarray, int]] = {}
        self._init_ack_cache()

    def _init_f5_ptbr(self):
        """Inicializa F5-TTS PT-BR apenas se explicitamente ativado via ENABLE_F5_GPU=true."""
        if os.getenv("ENABLE_F5_GPU", "false").lower() != "true":
            return
        try:
            import torchaudio
            from f5_tts.infer.utils_infer import (
                chunk_text,
                infer_batch_process,
                load_model,
                load_vocoder,
                preprocess_ref_audio_text,
            )
            from f5_tts.model import DiT
            from huggingface_hub import hf_hub_download

            self._f5_chunk_text = chunk_text
            self._f5_infer_batch = infer_batch_process
            self.f5_ref_tensor = None

            # Localiza no cache local primeiro para funcionar 100% offline sem requisição HTTP
            snap_glob = os.path.expanduser(
                "~/.cache/huggingface/hub/models--viperdf--f5-tts_pt-br/snapshots/*"
            )
            ckpts = sorted(glob.glob(os.path.join(snap_glob, "model_pt-br.safetensors")))
            vocabs = sorted(glob.glob(os.path.join(snap_glob, "vocab.txt")))

            ckpt_path = (
                ckpts[-1]
                if ckpts
                else hf_hub_download(repo_id="viperdf/f5-tts_pt-br", filename="model_pt-br.safetensors")
            )
            vocab_path = (
                vocabs[-1]
                if vocabs
                else hf_hub_download(repo_id="viperdf/f5-tts_pt-br", filename="vocab.txt")
            )

            model_cfg = dict(
                dim=1024,
                depth=22,
                heads=16,
                ff_mult=2,
                text_dim=512,
                text_mask_padding=False,
                conv_layers=4,
                pe_attn_head=1,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                self.f5_vocoder = load_vocoder(vocoder_name="vocos", is_local=False, device=self.device)
                self.f5_model = load_model(
                    DiT,
                    model_cfg,
                    ckpt_path,
                    mel_spec_type="vocos",
                    vocab_file=vocab_path,
                    ode_method="euler",
                    use_ema=True,
                    device=self.device,
                )

            # Prompt limpo de estúdio PT-BR (24kHz pré-RVC para zero degradação recursiva de consoantes)
            ref_candidates = [
                (
                    os.path.join(self.project_root, "models", "voice_prompts", "ptbr_clean_studio_ref.wav"),
                    "bom dia, mestre. todos os sistemas operacionais estão ativos.",
                ),
                (
                    os.path.join(self.project_root, "training", "f5_ptbr", "wavs", "aigis_pt_aigis_0004.wav"),
                    "bom dia! todos os módulos de vigilância e auxílio estão ativos.",
                ),
                (
                    os.path.join(
                        self.project_root, "training", "tts_distillation", "dataset", "wavs", "aigis_0004.wav"
                    ),
                    "bom dia! todos os módulos de vigilância e auxílio estão ativos.",
                ),
            ]
            for ref_wav, ref_txt in ref_candidates:
                if os.path.exists(ref_wav):
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.f5_ref_audio, self.f5_ref_text = preprocess_ref_audio_text(
                            ref_wav, ref_txt, show_info=lambda *a, **kw: None
                        )
                    self.f5_ref_tensor = torchaudio.load(self.f5_ref_audio)
                    break

            if self.f5_ref_tensor is not None:
                # Warmup FP16 silencioso para compilar kernels ROCm/AOTriton antes do loop de voz
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    _ = next(
                        self._f5_infer_batch(
                            self.f5_ref_tensor,
                            self.f5_ref_text,
                            ["olá, mestre."],
                            self.f5_model,
                            self.f5_vocoder,
                            mel_spec_type="vocos",
                            progress=None,
                            target_rms=0.08,
                            nfe_step=8,
                            cfg_strength=2.0,
                            sway_sampling_coef=-1.0,
                            speed=0.84,
                            device=self.device,
                        )
                    )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                logger.info(
                    "[VoiceSynthesizer] Motor Prosódico F5-TTS PT-BR (GPU FP16 + Vocos 24kHz) pronto e aquecido."
                )
        except Exception as e:
            logger.warning(f"[VoiceSynthesizer] Falha ao inicializar F5-TTS GPU (usando fallback): {e}")
            self.f5_model = None

    def _init_ack_cache(self):
        """Carrega arquivos WAV de ack pré-compilados em RAM para reprodução instantânea (<50ms)."""
        ack_dir = os.path.join(self.project_root, "output", "audio", "acks")
        os.makedirs(ack_dir, exist_ok=True)
        if os.path.exists(ack_dir):
            for fname in os.listdir(ack_dir):
                if fname.endswith(".wav"):
                    try:
                        key = os.path.splitext(fname)[0]
                        fpath = os.path.join(ack_dir, fname)
                        data, fs = sf.read(fpath, dtype="float32")
                        self._ack_cache[key] = (data, fs)
                    except Exception as e:
                        logger.debug(f"[VoiceSynthesizer] Falha ao carregar ack {fname}: {e}")

    def get_ack_audio(self, ack_type: str = "default") -> tuple[np.ndarray, int]:
        """Retorna a tupla (pcm, sr) do som de ack para enfileiramento síncrono sem race condition."""
        if not self._ack_cache:
            sr = 40000
            t = np.linspace(0, 0.08, int(sr * 0.08), endpoint=False)
            sine = (0.12 * np.sin(2 * np.pi * 880 * t) * np.exp(-t * 28)).astype(np.float32)
            return sine, sr

        chosen_key = ack_type if ack_type in self._ack_cache else next(iter(self._ack_cache))
        return self._ack_cache[chosen_key]

    async def play_immediate_ack(self, ack_type: str = "default") -> bool:
        """Reproduz um acknowledgement de áudio instantâneo serializado pelo lock de áudio."""
        data, fs = self.get_ack_audio(ack_type)
        await self.play_array(data, fs)
        return True

    def _detect_lang(self, text: str) -> str:
        """Heurística rápida de detecção pt vs en."""
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
        """
        Garante pronúncia integral de todas as palavras, numerais, siglas e nome 'Aigis'.
        """
        # Remove formatação Markdown (*, _, #, `) e parênteses de metadados
        clean = re.sub(r"[*_#`~()]+", " ", text)

        if lang == "en":
            return re.sub(r"\bAigis\b", "Eyegis", clean, flags=re.IGNORECASE).strip()

        # 1. 'Áiguis' gera o Hard G autêntico ('gu' como em Guilherme)
        clean = re.sub(r"\bAigis\b", "Áiguis", clean, flags=re.IGNORECASE)

        # 2. Anglicismos e termos técnicos frequentes em PT-BR para pronúncia natural nativa
        clean = _ANGLICISMS_PATTERN.sub(
            lambda m: _ANGLICISMS_LOOKUP[m.group(0).lower()], clean
        )

        # 2. Travessões isolados (" - " ou " — ") viram pausa natural de vírgula (preserva hífens em quinta-feira, servi-lo)
        clean = re.sub(r"\s+[-–—]+\s+", ", ", clean)

        # 3. Expande todos os números, horários e anos (ex: 5 -> cinco, 2026 -> dois mil e vinte e seis)
        clean = expand_numbers_pt_br(clean)

        # 5. Normaliza exclamações no meio de frases para ponto final (evita aceleração abrupta no F5-TTS)
        clean = re.sub(r"!\s+(?=[A-Za-zÀ-ÿ])", ". ", clean)

        # 6. Remove pontuações duplicadas e espaços múltiplos
        clean = re.sub(r"[!]{2,}", "!", clean)
        clean = re.sub(r"[?]{2,}", "?", clean)
        clean = re.sub(r"[.]{2,}", ".", clean)
        clean = re.sub(r"\s+", " ", clean).strip()

        if clean.endswith((",", ";", ":", "-")):
            clean = clean[:-1].strip() + "."
        elif clean and not clean.endswith((".", "!", "?")):
            clean += "."

        return clean

    async def _fetch_edge_tts(self, clean_text: str) -> tuple[np.ndarray, int] | None:
        """Sintetiza áudio via Edge-TTS (FranciscaNeural) 100% em memória RAM via WebSocket sem I/O de disco."""
        try:
            import edge_tts

            comm = edge_tts.Communicate(clean_text, self.voice_pt, rate="-4%", pitch="-1Hz")
            bio = io.BytesIO()
            async for ch in comm.stream():
                if ch["type"] == "audio":
                    bio.write(ch["data"])
            bio.seek(0)
            if bio.getbuffer().nbytes > 100:
                data, fs = sf.read(bio, dtype="float32")
                return data, fs
        except Exception as e:
            logger.warning(f"[VoiceSynthesizer] Edge-TTS indisponível ({e}), usando fallback local...")
        return None

    def _generate_base_pcm(
        self, text: str, fast_first_chunk: bool = False
    ) -> tuple[np.ndarray, int, str, float]:
        """Gera áudio base local em memória RAM/VRAM para offline ou fallback."""
        lang = self._detect_lang(text)
        clean_text = self._normalize_phonetics(text, lang)

        if lang == "en" and self.kokoro is not None:
            samples, sr = self.kokoro.create(
                clean_text, voice=self.kokoro_voice_en, speed=0.90, lang="en-us"
            )
            return samples.astype(np.float32), sr, lang, 0.0

        # 1. Fallback Piper Dii Local (leve e rápido)
        if self.pt_base_piper is not None:
            bio = io.BytesIO()
            with wave.open(bio, "wb") as wav_file:
                self.pt_base_piper.synthesize_wav(clean_text, wav_file, syn_config=self.syn_cfg)
            bio.seek(0)
            data, fs = sf.read(bio, dtype="float32")
            return data, fs, lang, 1.9

        # 2. Fallback F5-TTS GPU
        if lang == "pt" and self.f5_model is not None and self.f5_ref_tensor is not None:
            try:
                f5_prompt = clean_text.lower()
                if len(f5_prompt.encode("utf-8")) < 14:
                    f5_prompt = f5_prompt + " ..."
                speed = 0.88
                nfe_step = 10 if fast_first_chunk else 12
                batches = self._f5_chunk_text(f5_prompt, max_chars=135) or [f5_prompt]

                with self._gpu_lock, torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    wav_f5, sr_f5, _ = next(
                        self._f5_infer_batch(
                            self.f5_ref_tensor,
                            self.f5_ref_text,
                            batches,
                            self.f5_model,
                            self.f5_vocoder,
                            mel_spec_type="vocos",
                            progress=None,
                            target_rms=0.08,
                            cfg_strength=2.0,
                            sway_sampling_coef=-1.0,
                            nfe_step=nfe_step,
                            speed=speed,
                            device=self.device,
                        )
                    )
                return np.asarray(wav_f5, dtype=np.float32), int(sr_f5), lang, 0.0
            except Exception as e:
                logger.warning(f"[VoiceSynthesizer] Erro no F5-TTS GPU fallback: {e}")

        sr = 24000
        return np.zeros(int(sr * 0.3), dtype=np.float32), sr, lang, 0.0

    def convert_voice(
        self, input_wav_path: str, output_wav_path: str, pitch_shift: float = 0.0
    ) -> str:
        """Compatibilidade para conversão de arquivos WAV via vocal_tract em GPU."""
        data, sr = sf.read(input_wav_path, dtype="float32")
        with self._gpu_lock:
            out_pcm, out_sr = self.vocal_tract.convert_tensor(data, sr, pitch_shift=pitch_shift)
        sf.write(output_wav_path, out_pcm, out_sr)
        return output_wav_path

    async def synthesize_array(
        self, text: str, fast_first_chunk: bool = False
    ) -> tuple[np.ndarray, int]:
        """Sintetiza texto 100% em memória (Edge-TTS PT-BR ou Kokoro EN -> Trato Vocal RVC v2 40kHz GPU)."""
        lang = self._detect_lang(text)
        clean_text = self._normalize_phonetics(text, lang)

        # 1. Se for Piper destilado oficial
        if self.use_piper and lang == "pt" and self.piper_voice is not None:
            def _synth_piper():
                bio = io.BytesIO()
                with wave.open(bio, "wb") as wav_file:
                    self.piper_voice.synthesize_wav(clean_text, wav_file, syn_config=self.syn_cfg)
                bio.seek(0)
                return sf.read(bio, dtype="float32")

            return await asyncio.to_thread(_synth_piper)

        base_data = None
        base_sr = 24000
        pitch_shift = 0.0

        prefer_edge = os.getenv("PREFER_EDGE_TTS", "false").lower() == "true"

        # 2. Fonte Primária PT-BR:
        # Se prefer_edge for True, usa Edge-TTS com dicção articulada nativa.
        # Caso contrário, usa Piper Dii local em RAM para latência ultra-baixa (<150ms).
        if lang == "pt" and prefer_edge:
            edge_res = await self._fetch_edge_tts(clean_text)
            if edge_res is not None:
                base_data, base_sr = edge_res
                pitch_shift = 0.0

        if base_data is None and lang == "pt" and self.pt_base_piper is not None:
            def _get_piper_dii():
                bio = io.BytesIO()
                with wave.open(bio, "wb") as wav_file:
                    self.pt_base_piper.synthesize_wav(clean_text, wav_file, syn_config=self.syn_cfg)
                bio.seek(0)
                d, f = sf.read(bio, dtype="float32")
                return d, f, "pt", 1.9

            base_data, base_sr, _, pitch_shift = await asyncio.to_thread(_get_piper_dii)

        # 3. Fallback Edge-TTS se Piper local não estiver disponível
        elif base_data is None and lang == "pt" and os.getenv("DISABLE_EDGE_TTS", "false").lower() != "true":
            edge_res = await self._fetch_edge_tts(clean_text)
            if edge_res is not None:
                base_data, base_sr = edge_res
                pitch_shift = 0.0

        # 4. Fallback Geral (Kokoro para EN ou F5 se configurado)
        if base_data is None:
            def _get_base():
                return self._generate_base_pcm(clean_text, fast_first_chunk=fast_first_chunk)

            base_data, base_sr, _, pitch_shift = await asyncio.to_thread(_get_base)

        # 4. Trato Vocal Aigis RVC v2 40kHz Nativo em GPU
        # index_rate=0.15 e protect=0.55 garantem voz mais estável, eliminando aspereza e trêmulo
        def _convert_gpu():
            with self._gpu_lock:
                return self.vocal_tract.convert_tensor(
                    base_data,
                    base_sr,
                    pitch_shift=pitch_shift,
                    index_rate=0.15,
                    protect=0.55,
                    target_f0_med=225.0,
                )

        return await asyncio.to_thread(_convert_gpu)

    async def play_array(self, data: np.ndarray, fs: int):
        """
        Reproduz array de áudio serializado por Lock assíncrono com 65ms de tail-silence
        para garantir que o PipeWire/PortAudio drene 100% da última sílaba antes de fechar.
        """
        if data is None or len(data) == 0:
            return

        # Adiciona 65ms de silêncio final para evitar corte de cauda pelo PipeWire no Linux
        tail_pad = np.zeros(int(fs * 0.065), dtype=np.float32)
        padded = np.concatenate([data.astype(np.float32).flatten(), tail_pad])

        def _play_blocking():
            sd.play(padded, fs, blocking=True)

        async with self._play_lock:
            await asyncio.to_thread(_play_blocking)

    async def speak(
        self,
        text: str,
        output_filename: str = "aigis_live.wav",
        play_audio: bool = True,
    ) -> str:
        """Sintetiza fala da Aigis em GPU, salva cópia em disco e reproduz sem cortes."""
        out_pcm, out_sr = await self.synthesize_array(text)
        output_path = os.path.join(self.output_dir, output_filename)
        sf.write(output_path, out_pcm, out_sr)

        if play_audio:
            await self.play_array(out_pcm, out_sr)

        return output_path

    @staticmethod
    def play(wav_path: str, block: bool = True):
        data, fs = sf.read(wav_path, dtype="float32")
        tail_pad = np.zeros(int(fs * 0.065), dtype=np.float32)
        padded = np.concatenate([data.flatten(), tail_pad])
        sd.play(padded, fs, blocking=block)

    def close(self):
        """Encerra recursos e buffers residentes."""
        pass

    def __del__(self):
        self.close()


if __name__ == "__main__":
    synthesizer = VoiceSynthesizer()
    try:
        t0 = time.perf_counter()
        asyncio.run(
            synthesizer.speak(
                "Sistemas operacionais. Olá, mestre. Hoje é dia 5 de novembro de 2026.",
                "teste_rapido.wav",
                play_audio=False,
            )
        )
        print(f"Síntese completa em {(time.perf_counter() - t0)*1000:.1f}ms")
    finally:
        synthesizer.close()

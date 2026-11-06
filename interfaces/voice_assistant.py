import asyncio
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any

if sys.platform.startswith("linux"):
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
    os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
    os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

# Garante raiz do projeto no sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Garante suporte UTF-8 no terminal Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
from loguru import logger

from core.bus import CH_DISCORD_EVENTS, CH_PROACTIVE_ALERTS, EventBus
from core.client import chat
from core.stt import SpeechRecognizer
from core.timeline import SharedSession
from core.voice import VoiceSynthesizer

_BOUNDARY_RE = re.compile(r"([.!?]+|[,;:]|\n+|\))\s*")


def extract_ready_sentence(buffer_text: str, is_first_phrase: bool) -> tuple[str | None, str]:
    """
    Percorre todas as fronteiras de pontuação já seguidas de espaço em branco no buffer.
    Garante que:
    1. Nenhuma palavra BPE ou número decimal ('3.14') seja cortado ao meio.
    2. A primeira frase despache imediatamente com qualquer confirmação ('Entendido!', 'Boa noite!')
       para alcançar Time-To-First-Audio de ~1s.
    3. As orações seguintes agrupem contextos completos de 2+ palavras.
    """
    min_clause_words = 3 if is_first_phrase else 6

    for match in _BOUNDARY_RE.finditer(buffer_text):
        punct = match.group(1)
        candidate = buffer_text[: match.start() + len(punct)].strip()
        remainder = buffer_text[match.end() :]
        words = candidate.split()

        is_strong = any(p in punct for p in (".", "!", "?", "\n", ")"))

        if is_first_phrase:
            # Primeira frase: despacha imediatamente com 1+ palavras em pontuação forte ou 3+ em vírgula
            if is_strong and len(words) >= 1 and len(candidate) >= 3:
                return candidate, remainder
            if not is_strong and len(words) >= min_clause_words:
                clean_cand = candidate.rstrip(",;:- ").strip()
                if clean_cand and not clean_cand.endswith((".", "!", "?")):
                    clean_cand += "."
                return clean_cand, remainder
        else:
            # Orações seguintes: despacha em pontuação forte com 2+ palavras ou vírgula média (6+ palavras)
            if is_strong and len(words) >= 2:
                return candidate, remainder
            if not is_strong and len(words) >= min_clause_words:
                clean_cand = candidate.rstrip(",;:- ").strip()
                if clean_cand and not clean_cand.endswith((".", "!", "?")):
                    clean_cand += "."
                return clean_cand, remainder

    return None, buffer_text


async def play_audio_file(filepath: Path):
    """Toca o arquivo gerado via sounddevice."""
    try:
        import sounddevice as sd
        import soundfile as sf

        data, fs = sf.read(str(filepath), dtype="float32")
        sd.play(data, fs, blocking=True)
    except Exception as e:
        print(f"[Playback] Erro ao reproduzir áudio: {e}")


async def speak_notification(synth: VoiceSynthesizer, text: str):
    """Sintetiza e reproduz fala canônica da Aigis com o timbre autêntico de Persona 3."""
    print(f"\n🗣️ [Aigis Falando]: {text}")
    try:
        await synth.speak(text, play_audio=True)
    except Exception as e:
        print(f"[Aigis Falando] Erro na síntese: {e}")


async def main():
    print("=" * 60)
    print("🤖 SISTEMA DE VOZ INTEGRADO DA AIGIS (PC + DISCORD BRIDGE)")
    print("=" * 60)

    # 1. Carrega Sintetizador (100% GPU) e STT (Multi-thread CPU)
    synth = VoiceSynthesizer()
    stt = SpeechRecognizer(model_size="small", device="cpu", compute_type="int8")
    session = SharedSession.get_instance()

    # Registra o sintetizador para responder notificações de eventos externos (Discord, etc.)
    async def on_voice_notify(text: str, lang: str = "pt"):
        await speak_notification(synth, text)

    session.register_voice_notifier(on_voice_notify)

    # Conecta ao barramento Redis para consciência em tempo real de mensagens vindas do Discord
    bus = EventBus()

    async def on_discord_event(event: dict[str, Any]):
        author = event.get("author", "Usuário")
        content = event.get("content", "")
        channel = event.get("channel", "discord")
        if content:
            print(f"\n⚡ [Discord Live]: {author} em #{channel}: {content}")
            # Injeta a mensagem do Discord na linha temporal unificada
            await session.add_user_message(
                source=f"discord:#{channel}",
                content=f"[Mensagem enviada por {author} no Discord #{channel}]: {content}",
            )

    discord_listener_task = await bus.subscribe(CH_DISCORD_EVENTS, on_discord_event)

    # Fala inicial de prontidão
    print("\n[Inicializando Aigis...]")
    await speak_notification(
        synth, "Sistemas operacionais. Olá, mestre. Estou te ouvindo pelo microfone."
    )

    print("\n👉 Pressione Ctrl+C para encerrar o loop de voz.")
    print("👉 Fale normalmente no seu microfone. Quando você parar, a Aigis irá responder.\n")

    stop_event = threading.Event()

    try:
        while not stop_event.is_set():
            # Escuta uma elocução completa do usuário no microfone (timeout ágil de 1.0s)
            user_speech = await asyncio.to_thread(stt.listen_and_transcribe_once, 1.0, stop_event)
            if stop_event.is_set() or not user_speech:
                continue

            print(f'\n👤 [Natanael via Voz]: "{user_speech}"')

            sentence_queue: asyncio.Queue[tuple[str, bool] | None] = asyncio.Queue()
            audio_queue: asyncio.Queue[tuple[np.ndarray, int] | None] = asyncio.Queue()

            # Enfileira o Immediate Audio Ack como item #0 da própria fila de reprodução (zero race condition)
            await audio_queue.put(synth.get_ack_audio())

            t_first_audio = None

            async def synth_worker(
                s_queue: asyncio.Queue[tuple[str, bool] | None],
                a_queue: asyncio.Queue[tuple[np.ndarray, int] | None],
            ):
                while True:
                    item = await s_queue.get()
                    if item is None:
                        await a_queue.put(None)
                        s_queue.task_done()
                        break
                    frase, is_first = item
                    # 1. Filtra tags de ferramentas, JSON e código vazados do raciocínio interno
                    if re.search(r"<(tool_response|tool_call|function)|{\s*\"(success|returncode|stdout|stderr)|(execute_command|read_file|write_file|list_files|web_search)\s*\(", frase, re.IGNORECASE):
                        logger.warning(f"[VoiceAssistant] Sintetizador descartou vazamento interno: {frase[:60]}")
                        s_queue.task_done()
                        continue

                    # 2. Remove meta-comentários entre parênteses (ex: '(Executando comando localmente)')
                    frase_limpa = re.sub(r"^\s*\([^)]*(executando|comando|processando|localmente)[^)]*\)\s*", "", frase, flags=re.IGNORECASE).strip()
                    if not frase_limpa or len(frase_limpa) < 2 or not re.search(r"[a-zA-ZÀ-ÿ]{2,}", frase_limpa):
                        s_queue.task_done()
                        continue

                    try:
                        pcm, sr = await synth.synthesize_array(frase_limpa, fast_first_chunk=is_first)
                        await a_queue.put((pcm, sr))
                    except Exception as e:
                        logger.error(f"[VoiceAssistant] Erro ao sintetizar '{frase}': {e}")
                    finally:
                        s_queue.task_done()

            async def player_worker(
                a_queue: asyncio.Queue[tuple[np.ndarray, int] | None],
            ):
                nonlocal t_first_audio
                while True:
                    item = await a_queue.get()
                    if item is None:
                        a_queue.task_done()
                        break
                    pcm, sr = item
                    if t_first_audio is None and len(pcm) > 4000:
                        t_first_audio = (time.perf_counter() - t_llm_start) * 1000
                    try:
                        await synth.play_array(pcm, sr)
                    except Exception as e:
                        logger.error(f"[VoiceAssistant] Erro ao reproduzir áudio: {e}")
                    finally:
                        a_queue.task_done()

            synth_task = asyncio.create_task(synth_worker(sentence_queue, audio_queue))
            player_task = asyncio.create_task(player_worker(audio_queue))

            # Adiciona ao fluxo unificado de consciência
            await session.add_user_message(source="voice", content=user_speech)

            # Chama o cérebro com as mensagens sincronizadas (Discord + Voz) em Streaming por Sentença
            print("🧠 [Aigis processando...]")
            t_llm_start = time.perf_counter()
            tokens = []
            sync_messages = await session.get_synchronized_messages()

            async def handle_tool_feedback(func_name: str, func_args: dict):
                feedback_phrase = "Executando comando no sistema..."
                if func_name == "create_file":
                    feedback_phrase = "Criando o arquivo..."
                elif func_name == "delete_file":
                    feedback_phrase = "Apagando o arquivo..."
                elif func_name == "list_files":
                    feedback_phrase = "Listando arquivos..."
                elif func_name == "execute_command":
                    cmd = func_args.get("command", "")
                    if "Remove-Item" in cmd or "rm " in cmd or "del " in cmd:
                        feedback_phrase = "Apagando o arquivo..."
                    elif "Set-Content" in cmd or "New-Item" in cmd or "touch " in cmd:
                        feedback_phrase = "Criando o arquivo..."
                elif func_name == "web_search":
                    feedback_phrase = "Pesquisando na internet..."
                elif func_name == "read_file":
                    feedback_phrase = "Lendo arquivo..."

                print(f"⚡ [Feedback Imediato]: {feedback_phrase}")
                await sentence_queue.put((feedback_phrase, True))

            tools_executed = []
            buffer_texto = ""
            t_first_token = None
            is_first_phrase = True

            print("\n🗣️ [Aigis Falando]: ", end="", flush=True)

            async for token in chat(
                sync_messages,
                user_id=session.context.user.user_id,
                on_tool_start=handle_tool_feedback,
                executed_tools_collector=tools_executed,
            ):
                if t_first_token is None:
                    t_first_token = (time.perf_counter() - t_llm_start) * 1000

                tokens.append(token)
                buffer_texto += token
                print(token, end="", flush=True)

                while True:
                    ready_sentence, buffer_texto = extract_ready_sentence(
                        buffer_texto, is_first_phrase=is_first_phrase
                    )
                    if ready_sentence is None:
                        break
                    await sentence_queue.put((ready_sentence, is_first_phrase))
                    is_first_phrase = False

            # Fala qualquer resto pendente no buffer ao fim da geração
            resto = buffer_texto.strip().rstrip(",;:- ")
            if resto:
                if not resto.endswith((".", "!", "?", ":")):
                    resto += "."
                await sentence_queue.put((resto, is_first_phrase))

            dt_llm = (time.perf_counter() - t_llm_start) * 1000

            # Finaliza e aguarda as filas assíncronas de síntese e reprodução
            await sentence_queue.put(None)
            await synth_task
            await player_task

            print()
            dt_total_roundtrip = (time.perf_counter() - t_llm_start) * 1000
            resposta_completa = "".join(tokens).strip()
            if not resposta_completa:
                resposta_completa = "Comando processado com sucesso, mestre."

            if t_first_token is not None:
                print(f"⏱️ [Tempo até Primeiro Token LLM]: {t_first_token:.1f}ms")
            if t_first_audio is not None:
                print(f"⏱️ [Tempo até Início da Fala (TTFA)]: {t_first_audio:.1f}ms")
            print(f"⏱️ [Tempo de Geração LLM]: {dt_llm:.1f}ms")
            print(f"⏱️ [Duração Total do Turno]: {dt_total_roundtrip:.1f}ms")
            meta = {}
            if tools_executed:
                meta = {
                    "tool_calls": [item["tool_call"] for item in tools_executed],
                    "tool_results": [item["result"] for item in tools_executed],
                }
                # Notifica o Discord via Redis para registrar auditoria das ações
                for item in tools_executed:
                    t_call = item.get("tool_call", {})
                    fn = t_call.get("function", {}) if isinstance(t_call, dict) else {}
                    fn_name = fn.get("name", "ferramenta") if isinstance(fn, dict) else "ferramenta"
                    await bus.publish(
                        CH_PROACTIVE_ALERTS,
                        {
                            "type": "tool_executed",
                            "title": f"🛠️ Execução de Ferramenta via Voz: {fn_name}",
                            "description": f"A Aigis executou `{fn_name}` em resposta a comando de voz.",
                            "fields": {"Argumentos": str(fn.get("arguments", {}))[:300]},
                            "status": "success",
                        },
                    )
            await session.add_assistant_message(resposta_completa, metadata=meta)

    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        stop_event.set()
        print("\n[Encerrando processos de voz da Aigis...]")
        synth.close()
        print("Finalizado com sucesso.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nSessão encerrada.")

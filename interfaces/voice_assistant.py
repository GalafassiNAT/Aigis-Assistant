import asyncio
import sys
import time
from pathlib import Path

# Garante raiz do projeto no sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Garante suporte UTF-8 no terminal Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from core.client import chat
from core.stt import SpeechRecognizer
from core.timeline import SharedSession
from core.voice import VoiceSynthesizer


async def play_audio_file(filepath: Path):
    """Toca o arquivo gerado via sounddevice."""
    try:
        import sounddevice as sd
        import soundfile as sf

        data, fs = sf.read(str(filepath))
        sd.play(data, fs)
        sd.wait()
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

    # 1. Carrega Sintetizador e STT
    synth = VoiceSynthesizer()
    stt = SpeechRecognizer(model_size="small", device="cpu", compute_type="int8")
    session = SharedSession.get_instance()

    # Registra o sintetizador para responder notificações de eventos externos (Discord, etc.)
    async def on_voice_notify(text: str, lang: str = "pt"):
        await speak_notification(synth, text)

    session.register_voice_notifier(on_voice_notify)

    # Fala inicial de prontidão
    print("\n[Inicializando Aigis...]")
    await speak_notification(
        synth, "Sistemas operacionais! Olá mestre! Estou te ouvindo pelo microfone."
    )

    print("\n👉 Pressione Ctrl+C para encerrar o loop de voz.")
    print("👉 Fale normalmente no seu microfone. Quando você parar, a Aigis irá responder.\n")

    stop_event = threading.Event()

    try:
        while not stop_event.is_set():
            # Escuta uma elocução completa do usuário no microfone
            user_speech = await asyncio.to_thread(stt.listen_and_transcribe_once, 1.5, stop_event)
            if stop_event.is_set() or not user_speech:
                continue

            print(f'\n👤 [Natanael via Voz]: "{user_speech}"')

            # Adiciona ao fluxo unificado de consciência
            await session.add_user_message(source="voice", content=user_speech)

            # Chama o cérebro com as mensagens sincronizadas (Discord + Voz) em Streaming por Sentença
            print("🧠 [Aigis processando...]")
            t_llm_start = time.perf_counter()
            tokens = []
            sync_messages = await session.get_synchronized_messages()

            async def handle_tool_feedback(func_name: str, func_args: dict):
                feedback_phrase = "Executando comando no sistema..."
                if func_name == "execute_command":
                    cmd = func_args.get("command", "")
                    if "Remove-Item" in cmd or "rm " in cmd or "del " in cmd:
                        feedback_phrase = "Apagando o arquivo..."
                    elif "Set-Content" in cmd or "New-Item" in cmd:
                        feedback_phrase = "Criando o arquivo..."
                elif func_name == "web_search":
                    feedback_phrase = "Pesquisando na internet..."
                elif func_name == "read_file":
                    feedback_phrase = "Lendo arquivo..."

                print(f"⚡ [Feedback Imediato]: {feedback_phrase}")
                try:
                    await synth.speak(feedback_phrase, play_audio=True)
                except Exception:
                    pass

            tools_executed = []
            buffer_frase = []
            first_audio_played = False
            t_first_token = None

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
                buffer_frase.append(token)
                print(token, end="", flush=True)

                texto_acumulado = "".join(buffer_frase)
                # Dispara a síntese vocal imediatamente ao encontrar uma pontuação de fim de sentença
                if (
                    any(p in texto_acumulado for p in (".", "!", "?", "\n"))
                    and len(texto_acumulado.strip()) > 3
                ):
                    frase_para_falar = texto_acumulado.strip()
                    buffer_frase = []
                    await synth.speak(frase_para_falar, play_audio=True)
                    if not first_audio_played:
                        first_audio_played = True

            # Fala qualquer resto pendente no buffer
            resto = "".join(buffer_frase).strip()
            if resto:
                await synth.speak(resto, play_audio=True)

            print()
            dt_llm = (time.perf_counter() - t_llm_start) * 1000
            resposta_completa = "".join(tokens).strip()
            if not resposta_completa:
                resposta_completa = "Comando processado com sucesso, mestre."

            if t_first_token is not None:
                print(f"⏱️ [Tempo até Primeiro Token]: {t_first_token:.1f}ms")
            print(f"⏱️ [Tempo Total de Processamento]: {dt_llm:.1f}ms")
            meta = {}
            if tools_executed:
                meta = {
                    "tool_calls": [item["tool_call"] for item in tools_executed],
                    "tool_results": [item["result"] for item in tools_executed],
                }
            await session.add_assistant_message(resposta_completa, metadata=meta)

    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        stop_event.set()
        print("\n[Encerrando processos de voz da Aigis...]")
        synth.close()
        print("Finalizado com sucesso.")


if __name__ == "__main__":
    import threading

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nSessão encerrada.")

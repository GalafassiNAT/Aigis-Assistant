import asyncio

from core.client import chat, load_system_prompt
from core.context import ContextManager
from interfaces.formatters import LineStreamRenderer


async def main():
    context = ContextManager(max_chars=30000)
    context.set_system_prompt(load_system_prompt())

    print("=== Welcome to the chat! Type 'exit' to quit. ===\n")

    while True:
        user_input = await asyncio.to_thread(input, "You: ")

        if not user_input.strip():
            continue

        if user_input.lower() in ("exit", "sair", "quit"):
            print("\nBye bye!")
            break

        context.add_message("user", user_input)

        print("\nAssistant: ", end="", flush=True)
        response_tokens = []
        renderer = LineStreamRenderer()

        async for token in chat(context.get_messages(), user_id=context.user.user_id):
            response_tokens.append(token)
            out = renderer.feed(token)
            if out:
                print(out, end="", flush=True)

        resto = renderer.flush()
        if resto:
            print(resto, end="", flush=True)

        print("\n")
        context.add_message("assistant", "".join(response_tokens))


if __name__ == "__main__":
    asyncio.run(main())

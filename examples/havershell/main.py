import argparse
import inspect
import shutil
from dataclasses import dataclass

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import InMemoryHistory

from haverscript import Response, connect, echo


def bell():
    print("\a", end="", flush=True)  # Ring the bell


@dataclass(frozen=True)
class Commands:

    def help(self, session):
        print("/help")
        print("/bye")
        print("/context")
        print("/undo")
        print("/redo")
        return session

    def bye(self, _):
        exit()

    def context(self, session):
        print(session.render())
        return session

    def undo(self, session):
        if isinstance(session, Response):
            print("Undoing...")
            return session.parent
        bell()
        return session

    def redo(self, session):
        if isinstance(session, Response):
            print("Redoing...")
            # print prompt again?
            return session.redo()
        bell()
        return session


def main():
    models = connect().list()

    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Haverscript Shell")
    parser.add_argument(
        "--model",
        type=str,
        choices=models,
        required=True,
        help="The model to use for processing.",
    )
    parser.add_argument(
        "--context",
        type=str,
        help="The context to use for processing.",
    )
    parser.add_argument(
        "--cache",
        type=str,
        help="database to use as a cache.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="temperature of replies.",
    )
    parser.add_argument(
        "--num_predict",
        type=int,
        help="cap on number of tokens in reply.",
    )
    parser.add_argument("--num_ctx", type=int, help="context size")

    args = parser.parse_args()

    terminal_size = shutil.get_terminal_size(fallback=(80, 24))
    terminal_width = terminal_size.columns
    llm = connect(args.model) | echo(width=terminal_width - 2, prompt=False)

    if args.temperature is not None:
        llm = llm.options(temperature=args.temperature)
    if args.num_predict is not None:
        llm = llm.options(num_predict=args.num_predict)
    if args.num_ctx is not None:
        llm = llm.options(num_ctx=args.num_ctx)

    if args.cache:
        llm = llm.cache(args.cache)

    if args.context:
        with open(args.context) as f:
            session = llm.load(markdown=f.read())
    else:
        session = llm

    command_list = [
        "/" + method
        for method in dir(Commands)
        if callable(getattr(Commands, method)) and not method.startswith("__")
    ]

    print(f"connected to {args.model} (/help for help) ")

    while True:
        try:
            if previous := session.children():
                history = InMemoryHistory()
                for prompt in previous.prompt:
                    history.append_string(prompt)

                prompt_session = PromptSession(history=history)
            else:
                prompt_session = PromptSession()

            completer = WordCompleter(command_list, sentence=True)

            try:
                text = prompt_session.prompt("> ", completer=completer)
            except EOFError:
                exit()

            if text.startswith("/"):
                cmd = text.split(maxsplit=1)[0]
                if cmd in command_list:
                    after = text[len(cmd) :].strip()
                    function = getattr(Commands(), cmd[1:], None)
                    if len(inspect.signature(function).parameters) > 1:
                        if not after:
                            print(f"{cmd} expects an argument")
                            continue
                        session = function(session, after)
                    else:
                        if after:
                            print(f"{cmd} does not take any argument")
                            continue
                        session = function(session)
                    continue

                print(f"Unknown command {text}.")
                continue
            else:
                print()
                try:
                    session = session.chat(text)
                except KeyboardInterrupt:
                    print("^C\n")
                print()
        except KeyboardInterrupt:
            print("Use Ctrl + d or /bye to exit.")


if __name__ == "__main__":
    main()

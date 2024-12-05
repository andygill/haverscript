import re
import threading
from dataclasses import dataclass

from yaspin import yaspin

from .languagemodel import LanguageModel, LanguageModelResponse


@dataclass(frozen=True)
class Middleware(LanguageModel):
    """Middleware is a LanguageModel that has next down-the-pipeline LanguageModel."""

    next: LanguageModel


@dataclass(frozen=True)
class ModelMiddleware(Middleware):
    model: str

    def chat(self, prompt: str, **kwargs):
        return self.next.chat(prompt, model=self.model, **kwargs)


@dataclass(frozen=True)
class EchoMiddleware(Middleware):
    width: int = 78
    prompt: bool = True

    def chat(self, prompt: str, **kwargs):
        if self.prompt:
            print()
            print("\n".join([f"> {line}" for line in prompt.splitlines()]))
            print()

        # We turn on streaming, because if we echo, we want to see progress
        response = self.next.chat(prompt, stream=True, **kwargs)

        assert isinstance(response, LanguageModelResponse)

        tokens = self._wrap((token for token in response if isinstance(token, str)))

        first_token_available = threading.Event()
        first_token = [None]  # Use list to allow modification in nested scope

        def get_first_token():
            try:
                first_token[0] = next(tokens)
            except Exception as e:
                first_token[0] = e
            finally:
                first_token_available.set()

        threading.Thread(target=get_first_token).start()

        with yaspin() as spinner:
            first_token_available.wait()

        if first_token[0] is not None:
            if isinstance(first_token[0], Exception):
                raise first_token[0]
            print(first_token[0], end="", flush=True)

        for token in tokens:
            print(token, end="", flush=True)
        print()  # finish with a newline

        return response

    def _wrap(self, stream):

        line_width = 0  # the size of the commited line so far
        spaces = 0
        newline = "\n"
        prequel = ""

        for s in stream:

            for t in re.split(r"(\n|\S+)", s):

                if t == "":
                    continue

                if t.isspace():
                    if line_width + spaces + len(prequel) > self.width:
                        yield newline  # injected newline
                        line_width = 0
                        spaces = 0

                    if spaces > 0 and line_width > 0:
                        yield " " * spaces
                        line_width += spaces

                    spaces = 0
                    line_width += spaces + len(prequel)
                    yield prequel
                    prequel = ""

                    if t == "\n":
                        line_width = 0
                        yield newline  # actual newline
                    else:
                        spaces += len(t)
                else:
                    prequel += t

        if prequel != "":
            if line_width + spaces + len(prequel) > self.width:
                yield newline
                line_width = 0
                spaces = 0

            if spaces > 0 and line_width > 0:
                yield " " * spaces

            yield prequel

        return

    def list(self):
        return []

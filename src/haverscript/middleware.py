import re
import threading
import queue
import time
import json
import os
from datetime import datetime

from dataclasses import dataclass
import queue
import time
from typing import AnyStr, Callable, Optional, Self, Tuple, NoReturn

from tenacity import Retrying, RetryError
from yaspin import yaspin

from .exceptions import LLMResultError, LLMError
from .languagemodel import LanguageModel, LanguageModelResponse
from .cache import Cache
from .render import *


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
class RetryMiddleware(Middleware):
    """Retry the lower chat if it fails, using options as argument(s) to tenacity retry.

    There is a cavet here. If the lower chat returns an in-progress and working streaming,
    then this will be accepted by this retry. We wait for the first token, though.
    """

    options: dict

    def chat(self, prompt: str, **kwargs):
        try:
            for attempt in Retrying(**self.options):
                with attempt:
                    # We turn off streaming, because we want complete results.
                    return self.next.chat(prompt, streaming=False, **kwargs)
        except RetryError as e:
            raise LLMResultError()


@dataclass(frozen=True)
class ValidationMiddleware(Middleware):
    """Validate if a predicate is true for the response."""

    predicate: Callable[[str], bool]

    def chat(self, prompt: str, **kwargs):
        response = self.next.chat(prompt, **kwargs)
        # the str forces the full evaluation here
        if not self.predicate(str(response)):
            raise LLMResultError()
        return response


@dataclass(frozen=True)
class EchoMiddleware(Middleware):
    width: int = 78
    prompt: bool = True
    spinner: bool = True

    def chat(self, prompt: str, **kwargs):
        if prompt is None:
            return self.next.chat(prompt, **kwargs)

        if self.prompt and prompt:
            print()
            print("\n".join([f"> {line}" for line in prompt.splitlines()]))
            print()

        # We turn on streaming to make the echo responsive
        kwargs["stream"] = True

        if self.spinner:
            event = threading.Event()

            def wait_for_event():
                with yaspin() as spinner:
                    event.wait()  # Wait until the event is set

            spinner_thread = threading.Thread(target=wait_for_event)
            spinner_thread.start()

            try:
                response = self.next.chat(prompt, **kwargs)
            finally:
                # stop the spinner
                event.set()
                # wait for the spinner thread to stop (and stop printing)
                spinner_thread.join()
        else:
            response = self.next.chat(prompt, **kwargs)

        assert isinstance(response, LanguageModelResponse)

        for token in self._wrap(response.tokens()):
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


@dataclass(frozen=True)
class StatsMiddleware(Middleware):
    width: int = 78
    prompt: bool = True

    def chat(self, prompt: str, **kwargs):

        event = threading.Event()
        channel = queue.Queue()

        start_time = time.time()

        def message(prompt, tokens, time_to_first_token, tokens_per_second):
            return (
                f"prompt : {len(prompt):,}b, "
                f"reply : {tokens:,}t, "
                f"first token : {time_to_first_token:.2f}s, "
                f"tokens/s : {tokens_per_second:.0f}"
            )

        def wait_for_event():
            with yaspin(text=f"prompt : {len(prompt):,}b") as spinner:
                first_token_time = None
                tokens = 0
                while True:
                    token = channel.get()
                    if not isinstance(token, str):
                        txt = spinner.text
                        if isinstance(token, LLMError):
                            txt += ", LLMError Exception raised"
                        spinner.write("- " + txt)
                        break

                    now = time.time()
                    if first_token_time is None:
                        first_token_time = now
                        time_to_first_token = first_token_time - start_time
                        tokens_per_second = 0
                    else:
                        tokens_per_second = tokens / (now - first_token_time)
                    tokens += 1

                    spinner.text = message(
                        prompt, tokens, time_to_first_token, tokens_per_second
                    )

        try:
            spinner_thread = threading.Thread(target=wait_for_event)
            spinner_thread.start()

            # We turn on streaming to make the echo responsive
            kwargs["stream"] = True
            response = self.next.chat(prompt, **kwargs)
            assert isinstance(response, LanguageModelResponse)

            for token in response.tokens():
                channel.put(token)
        except LLMError as e:
            channel.put(e)
            spinner_thread.join()  # do wait for the printing to finish
            raise e
        finally:
            channel.put(None)

        # wait for the spinner thread to stop (and stop printing)
        spinner_thread.join()

        return response


@dataclass(frozen=True)
class CacheMiddleware(Middleware):

    filename: str
    mode: str  # "r", "a", "a+"

    def chat(
        self,
        prompt: str,
        system: str,
        context: str,
        images: list,
        options: dict,
        **kwargs,
    ):

        cache = Cache(self.filename, self.mode)

        parameters = dict(options)
        fresh = "fresh" in parameters and parameters["fresh"]
        if fresh:
            # fresh is about ignoring cache misses, and nothing else.
            # So we do not record if this cache write was triggered
            # by a previous cache miss, or a request with fresh.
            del parameters["fresh"]

        if self.mode in {"r", "a+"} and not fresh:

            cached = cache.lookup_interactions(
                system, context, prompt, images, parameters, limit=1, blacklist=True
            )

            if cached:
                key = next(iter(cached.keys()))
                if self.mode == "a+":
                    cache.blacklist(key)
                # just return the (cached) reply
                return LanguageModelResponse(cached[key][2])

        response = self.next.chat(
            prompt=prompt,
            system=system,
            context=context,
            images=images,
            options=options,
            **kwargs,
        )
        if self.mode == "r":
            return response

        def save_response():
            cache.insert_interaction(
                system, context, prompt, images, str(response), parameters
            )

        response.after(save_response)

        return response

    def children(
        self,
        prompt: str,
        system: str,
        context: str,
        images: list,
        options: dict,
        **kwargs,
    ):
        if self.mode == "a":
            return []

        cache = Cache(self.filename, self.mode)

        parameters = options.copy()

        return cache.lookup_interactions(
            system, context, prompt, [], parameters, limit=None, blacklist=False
        ).values()


@dataclass(frozen=True)
class TranscriptMiddleware(Middleware):
    dirname: str

    def chat(self, prompt: str, system: str | None, context: tuple, **kwargs):
        response: LanguageModelResponse = self.next.chat(
            prompt=prompt, system=system, context=context, **kwargs
        )

        dirname = self.dirname
        # Ensure the parent directory exists
        if not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)

        transcript = render_system(system)

        for p, _, r in context:
            transcript = render_interaction(transcript, p, r)

        def write_transcript():
            transcript_file = datetime.now().strftime("%Y%m%d_%H:%M:%S.%f.md")
            transcript_ = render_interaction(transcript, prompt, str(response))
            with open(os.path.join(dirname, transcript_file), "w") as file:
                file.write(transcript_)

        response.after(write_transcript)
        return response

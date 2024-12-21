import re
import threading
import queue
import time
import json
import os
from datetime import datetime
from abc import abstractmethod
import builtins
from copy import deepcopy
from pydantic import BaseModel

from dataclasses import dataclass, field
import queue
import time
from typing import AnyStr, Callable, Optional, Self, Tuple, NoReturn

from tenacity import Retrying, RetryError
from yaspin import yaspin

from .exceptions import LLMResultError, LLMError
from .languagemodel import (
    LanguageModel,
    Reply,
    Request,
    Informational,
    Exchange,
)
from .cache import Cache
from .render import *
from functools import partial
from abc import ABC


@dataclass(frozen=True)
class Middleware(ABC):
    """Middleware is a bidirectional Prompt and Reply processor.

    Middleware is something you use on a LanguageModel,
    and a LanguageModel is something you *call*.
    """

    @abstractmethod
    def invoke(self, request: Request, next: LanguageModel) -> Reply:
        return next.ask(request=request)

    def first(self) -> Self | None:
        """get the first Middleware in the pipeline (from the Prompt's point of view)"""
        return self

    def __or__(self, other: LanguageModel) -> LanguageModel:
        return AppendMiddleware(self, other)


@dataclass(frozen=True)
class MiddlewareLanguageModel(LanguageModel):
    """MiddlewareLanguageModel is Middleware with a specific target LanguageModel.

    This combination of is itself a LanguageModel.
    """

    middleware: Middleware
    next: LanguageModel

    def ask(self, request: Request) -> Reply:
        return self.middleware.invoke(request=request, next=self.next)


@dataclass(frozen=True)
class AppendMiddleware(Middleware):
    after: Middleware
    before: Middleware  # we evaluate from right to left in invoke

    def invoke(self, request: Request, next: LanguageModel) -> Reply:
        # The ice is thin here but it holds.
        return self.before.invoke(
            request=request, next=MiddlewareLanguageModel(self.after, next)
        )

    def first(self) -> Self:
        if first := self.before.first():
            return first
        return self.after.first()


@dataclass(frozen=True)
class EmptyMiddleware(Middleware):

    def invoke(self, request: Request, next: LanguageModel) -> Reply:
        return next.ask(request=request)


@dataclass(frozen=True)
class RetryMiddleware(Middleware):
    """Retry the lower chat if it fails, using options as argument(s) to tenacity retry.

    There is a cavet here. If the lower chat returns an in-progress and working streaming,
    then this will be accepted by this retry. We wait for the first token, though.
    """

    options: dict

    def invoke(self, request: Request, next: LanguageModel):
        try:
            for attempt in Retrying(**self.options):
                with attempt:
                    return next.ask(request=request)
        except RetryError as e:
            print(e)
            raise LLMResultError()


def retry(**options) -> Self:
    """retry uses tenacity to wrap the LLM request-response action in retry options."""
    return RetryMiddleware(options)


@dataclass(frozen=True)
class ValidationMiddleware(Middleware):
    """Validate if a predicate is true for the response."""

    predicate: Callable[[str], bool]

    def invoke(self, request: Request, next: LanguageModel):
        response = next.ask(request=request)
        # the str forces the full evaluation here
        if not self.predicate(str(response)):
            raise LLMResultError()
        return response


def validate(predicate: Callable[[str], bool]):
    return ValidationMiddleware(predicate)


@dataclass(frozen=True)
class EchoMiddleware(Middleware):
    width: int = 78
    prompt: bool = True
    spinner: bool = True

    def invoke(self, request: Request, next: LanguageModel):

        prompt = request.prompt

        if prompt is None:
            return next.ask(request=request)

        if self.prompt and prompt:
            print()
            print("\n".join([f"> {line}" for line in prompt.splitlines()]))
            print()

        # We turn on streaming to make the echo responsive
        request = request.model_copy(update=dict(stream=True))

        channel = queue.Queue()
        back_channel = queue.Queue()

        def wait_for_event():
            loop = True
            while loop:
                try:
                    # we wait a fraction of a second before reading the message
                    message = channel.get(timeout=0.1)
                except queue.Empty:
                    with yaspin() as spinner:
                        while True:
                            message = channel.get()
                            if message == "done" or message == "stop":
                                break
                            spinner.text = message + " "
                if message == "done":
                    return
                back_channel.put("waiting")
                while True:
                    message = channel.get()
                    if message == "start":
                        break
                    if message == "done":
                        return

        try:
            if self.spinner:
                spinner_thread = threading.Thread(target=wait_for_event)
                spinner_thread.start()

            response: Reply = next.ask(request=request)

            newline = True
            for token in self._wrap(response):
                if isinstance(token, str):
                    if newline and self.spinner:
                        channel.put("stop")
                        back_channel.get()
                    print(token, end="", flush=True)
                    newline = token.endswith("\n")
                    if newline and self.spinner:
                        channel.put("start")
                if isinstance(token, Informational):
                    channel.put(token.message)
        finally:
            if self.spinner:
                channel.put("done")
                spinner_thread.join()

        print()  # finish with a newline

        return response

    def _wrap(self, stream):

        line_width = 0  # the size of the commited line so far
        spaces = 0
        newline = "\n"
        prequel = ""

        for s in stream:

            if isinstance(s, Informational):
                s = s.message

            if not isinstance(s, str):
                yield s
                continue

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


def echo(width: int = 78, prompt: bool = True, spinner: bool = True) -> Middleware:
    """echo prompts and responses to stdout."""
    assert isinstance(width, int) and not isinstance(width, bool)
    assert isinstance(prompt, bool)
    assert isinstance(spinner, bool)

    return EchoMiddleware(width, prompt, spinner)


@dataclass(frozen=True)
class StatsMiddleware(Middleware):
    width: int = 78
    prompt: bool = True

    def invoke(self, request: Request, next: LanguageModel) -> Reply:

        prompt = request.prompt
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
            request = request.model_copy(update=dict(stream=True))
            response = next.ask(request=request)
            assert isinstance(response, Reply)

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


def stats() -> Middleware:
    return StatsMiddleware()


@dataclass(frozen=True)
class CacheMiddleware(Middleware):

    filename: str
    mode: str  # "r", "a", "a+"

    def invoke(self, request: Request, next: LanguageModel):

        cache = Cache(self.filename, self.mode)

        parameters = dict(request.contexture.options)
        # TODO: make this Request parameter

        if self.mode in {"r", "a+"} and not request.fresh:

            cached = cache.lookup_interactions(
                request.contexture.system,
                request.contexture.context,
                request.prompt,
                request.images,
                parameters,
                limit=1,
                blacklist=True,
            )

            if cached:
                key = builtins.next(iter(cached.keys()))
                if self.mode == "a+":
                    cache.blacklist(key)
                # just return the (cached) reply
                return Reply(cached[key][2])

        response = next.ask(request=request)
        if self.mode == "r":
            return response

        def save_response():
            cache.insert_interaction(
                request.contexture.system,
                request.contexture.context,
                request.prompt,
                request.images,
                str(response),
                parameters,
            )

        response.after(save_response)

        return response

    def children(self, request: Request):
        if self.mode == "a":
            return []

        prompt = request.prompt
        system = request.contexture.system
        context = request.contexture.context
        images = request.images
        options = request.contexture.options

        cache = Cache(self.filename, self.mode)

        parameters = options.copy()

        return cache.lookup_interactions(
            system, context, prompt, [], parameters, limit=None, blacklist=False
        ).values()


def cache(filename: str, mode: str | None = "a+") -> Middleware:
    """Set the cache filename for this model."""
    return CacheMiddleware(filename, mode)


@dataclass(frozen=True)
class TranscriptMiddleware(Middleware):
    dirname: str

    def invoke(self, request: Request, next: LanguageModel) -> Reply:

        response: Reply = next.ask(request=request)

        dirname = self.dirname
        # Ensure the parent directory exists
        if not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)

        transcript = render_system(request.contexture.system)

        for exchange in request.contexture.context:
            transcript = render_interaction(transcript, exchange.prompt, exchange.reply)

        def write_transcript():
            transcript_file = datetime.now().strftime("%Y%m%d_%H:%M:%S.%f.md")
            transcript_ = render_interaction(transcript, request.prompt, str(response))
            with open(os.path.join(dirname, transcript_file), "w") as file:
                file.write(transcript_)

            latest_symlink = os.path.join(dirname, "latest.md")

            if os.path.islink(latest_symlink):
                os.unlink(latest_symlink)

            os.symlink(transcript_file, latest_symlink)

        response.after(write_transcript)
        return response


def transcript(dirname: str):
    """write a full transcript of every interaction, in a subdirectory."""
    return TranscriptMiddleware(dirname)


@dataclass(frozen=True)
class FreshMiddleware(Middleware):

    def invoke(self, request: Request, next: LanguageModel) -> Reply:
        request = request.model_copy(update=dict(fresh=True))
        return next.ask(request=request)


def fresh():
    """require any cached reply be ignored, and a fresh reply be generated."""
    return FreshMiddleware()


@dataclass(frozen=True)
class ModelMiddleware(Middleware):
    model: str

    def invoke(self, request: Request, next: LanguageModel) -> Reply:
        contexture = request.contexture.model_copy(update=dict(model=self.model))
        request = request.model_copy(update=dict(contexture=contexture))
        return next.ask(request=request)


def model(model: str) -> Middleware:
    return ModelMiddleware(model)


@dataclass(frozen=True)
class OptionsMiddleware(Middleware):
    options: dict

    def invoke(self, request: Request, next: LanguageModel) -> Reply:
        contexture = request.contexture.model_copy(
            update=dict(options=self.options | request.contexture.options)
        )
        request = request.model_copy(update=dict(contexture=contexture))
        return next.ask(request=request)


def options(**kwargs) -> Middleware:
    """Options to pass to the model, such as temperature and seed.

    num_ctx: int
    num_keep: int
    seed: int
    num_predict: int
    top_k: int
    top_p: float
    tfs_z: float
    typical_p: float
    repeat_last_n: int
    temperature: float
    repeat_penalty: float
    presence_penalty: float
    frequency_penalty: float
    mirostat: int
    mirostat_tau: float
    mirostat_eta: float
    penalize_newline: bool
    stop: Sequence[str]
    """
    return OptionsMiddleware(kwargs)


class MetaModel(BaseModel):
    system: str | None

    @abstractmethod
    def chat(self, prompt, next: LanguageModel) -> Reply:
        """Turn a chat-with-prompt into a follow-on call of the next model."""


@dataclass(frozen=True)
class MetaMiddleware(Middleware):
    model: MetaModel

    _model_cache: dict = field(init=False, default_factory=dict)

    def invoke(self, request: Request, next: LanguageModel) -> Reply:
        system = request.contexture.system
        context = request.contexture.context

        try:
            model = self._model_cache[system, context]
        except KeyError:
            if context == ():
                model = self.model(system=system)
                self._model_cache[system, ()] = model
            else:
                # We has a context we've never seen
                # Which means we did not generate it
                # Which means we reject it
                assert False, "unknown system or context"

        model: MetaModel = deepcopy(model)

        response: Reply = model.chat(request.prompt, next)

        def after():
            exchange = Exchange(prompt=request.prompt, images=(), reply=str(response))
            self._model_cache[system, context + (exchange,)] = model

        response.after(after)
        return response


def meta(model: MetaModel):
    return MetaMiddleware(model)

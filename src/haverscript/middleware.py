from __future__ import annotations

import builtins
import json
import logging as log
import os
import queue
import re
import textwrap
import threading
import time
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Type, get_origin, get_args, Union
import types
import jsonref

from pydantic import BaseModel, TypeAdapter
from yaspin import yaspin

from .cache import Cache
from .exceptions import LLMError, LLMResultError
from .types import (
    Exchange,
    Informational,
    LanguageModel,
    Reply,
    Request,
    Value,
    Middleware,
    EmptyMiddleware,
    Exchange,
    AssistantMessage,
    Prompt,
    ToolCall,
)
from .render import *

logger = log.getLogger("haverscript")


@dataclass(frozen=True)
class RetryMiddleware(Middleware):
    """Retry the lower chat if it fails.

    There is a cavet here. If the lower chat returns an in-progress and working streaming,
    then this will be accepted by this retry. We wait for the first token, though.

    We need to increment the seed, if any, each time.
    """

    count: dict

    def invoke(self, request: Request, next: LanguageModel):
        seed = None
        for _ in range(self.count):
            if seed:
                request.model_copy(
                    update=dict(
                        contexture=self.contexture.model_copy(
                            update=dict(
                                options=request.contexture.options | dict(seed=seed)
                            )
                        )
                    )
                )
            try:
                return next.ask(request=request)
            except Exception as e:
                # we caught the exception, and now retry after 1 second
                time.sleep(1)
            finally:
                seed = (
                    request.contexture.options["seed"] + 1
                    if "seed" in request.contexture.options
                    else None
                )

        raise LLMResultError()


def retry(count: int) -> Middleware:
    """retry allows multiple attempts at a specfic LLM call."""
    return RetryMiddleware(count=count)


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


def validate(predicate: Callable[[str], bool]) -> Middleware:
    """validate the response as middleware. Can raise as LLMResultError"""
    return ValidationMiddleware(predicate)


@dataclass(frozen=True)
class EchoMiddleware(Middleware):
    width: int = 78
    prompt: bool = True
    spinner: bool = True
    stream: bool = True

    def invoke(self, request: Request, next: LanguageModel):

        prompt = request.prompt

        if prompt is None:
            return next.ask(request=request)

        if self.prompt and prompt and isinstance(prompt, Prompt):
            print()
            print("\n".join([f"> {line}" for line in prompt.content.splitlines()]))
            print()

        # We turn on streaming to make the echo responsive
        request = request.model_copy(update=dict(stream=self.stream))

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
                if isinstance(token, (str, ToolCall)):
                    if newline and self.spinner:
                        channel.put("stop")
                        back_channel.get()
                    if isinstance(token, ToolCall):
                        formatted_args = ", ".join(
                            f"{key} = {repr(value)}"
                            for key, value in token.arguments.items()
                        )
                        print(f"Calling {token.name}({formatted_args})")
                        newline = True
                    else:
                        print(token, end="", flush=True)
                        newline = token.endswith("\n")
                    if newline and self.spinner:
                        channel.put("start")
                if isinstance(token, Informational):
                    channel.put(token.message)
                if isinstance(token, ToolCall):
                    channel.put(str(token))
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


def echo(
    width: int = 78, prompt: bool = True, spinner: bool = True, stream: bool = True
) -> Middleware:
    """echo prompts and responses to stdout.

    if prompt=False, do not echo the prompt
    if spinner=False, do not use a spinner. (For example, when redirecting to a file)
    if stream=False, do not use streaming.
    """
    assert isinstance(width, int) and not isinstance(width, bool)
    assert isinstance(prompt, bool)
    assert isinstance(spinner, bool)

    return EchoMiddleware(width, prompt, spinner, stream)


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
                f"prompt : {len(prompt.content):,}b, "
                f"reply : {tokens:,}t, "
                f"first token : {time_to_first_token:.2f}s, "
                f"tokens/s : {tokens_per_second:.0f}"
            )

        def wait_for_event():
            with yaspin(text=f"prompt : {len(prompt.content):,}b") as spinner:
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
    """print stats to stdout."""
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
                request.prompt.content,
                request.prompt.images,
                parameters,
                limit=1,
                blacklist=True,
            )

            if cached:
                key = builtins.next(iter(cached.keys()))
                if self.mode == "a+":
                    cache.blacklist(key)
                # just return the (cached) reply
                return Reply([cached[key][2]])

        response = next.ask(request=request)
        if self.mode == "r":
            return response

        def save_response():
            cache.insert_interaction(
                request.contexture.system,
                request.contexture.context,
                request.prompt.content,
                request.prompt.images,
                str(response),
                parameters,
            )

        response.after(save_response)

        return response

    def children(self, request: Request):
        if self.mode == "a":
            return []

        if request.prompt is not None:
            prompt = request.prompt.content
        else:
            prompt = None
        system = request.contexture.system
        context = request.contexture.context
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
            transcript = render_interaction(transcript, exchange)

        def write_transcript():
            transcript_file = datetime.now().strftime("%Y%m%d_%H:%M:%S.%f.md")
            transcript_ = render_interaction(
                transcript,
                Exchange(
                    prompt=request.prompt,
                    reply=AssistantMessage(content=str(response)),
                ),
            )
            with open(os.path.join(dirname, transcript_file), "w") as file:
                file.write(transcript_)

            latest_symlink = os.path.join(dirname, "latest.md")

            if os.path.islink(latest_symlink):
                os.unlink(latest_symlink)

            os.symlink(transcript_file, latest_symlink)

        response.after(write_transcript)
        return response


def transcript(dirname: str) -> Middleware:
    """write a full transcript of every interaction, in a subdirectory."""
    return TranscriptMiddleware(dirname)


@dataclass(frozen=True)
class TraceMiddleware(Middleware):
    level: int = log.DEBUG

    def invoke(self, request: Request, next: LanguageModel) -> Reply:

        logger.log(self.level, f"request={repr(request)}")

        reply: Reply = next.ask(request=request)

        # we give the reply twice, once when we first get it,
        # and second after the reply is complete.
        logger.log(self.level, f"initial reply={repr(reply)}")

        def after():
            logger.log(self.level, f"final reply={repr(reply)}")

        reply.after(after)

        return reply


def trace(level: int = log.DEBUG) -> Middleware:
    """Log all requests and responses."""
    return TraceMiddleware(level)


@dataclass(frozen=True)
class RequestOptionMiddleware(Middleware):
    fresh: bool | None = None
    stream: bool | None = None
    speed: int | None = None

    def invoke(self, request: Request, next: LanguageModel) -> Reply:
        options = {}
        if self.fresh is not None:
            options["fresh"] = self.fresh
        if self.stream is not None:
            options["stream"] = self.stream

        request = request.model_copy(update=options)
        reply = next.ask(request=request)
        if self.speed is None:
            return reply

        def streaming():
            ts0 = time.monotonic()
            for block in reply:
                if isinstance(block, str):
                    for token in re.findall(r"\S+|\s+", block):
                        ts1 = time.monotonic()
                        remaining = (1 / self.speed) - (ts1 - ts0)
                        ts0 = ts1
                        if remaining > 0:
                            time.sleep(remaining)
                        yield token
                else:
                    yield block

        return Reply(streaming())


def fresh() -> Middleware:
    """require any cached reply be ignored, and a fresh reply be generated."""
    return RequestOptionMiddleware(fresh=True)


def stream(speed: int | None = None) -> Middleware:
    """turn on streaming for LLM response.

    The speed is the frequency of tokens, per second,
    defaulting to as-fast-as-possible.

    speed can be used to simulate slower models,
    and slow down cache hits to simulate model token speed.
    """
    return RequestOptionMiddleware(stream=True, speed=speed)


@dataclass(frozen=True)
class ModelMiddleware(Middleware):
    model: str

    def invoke(self, request: Request, next: LanguageModel) -> Reply:
        contexture = request.contexture.model_copy(update=dict(model=self.model))
        request = request.model_copy(update=dict(contexture=contexture))
        return next.ask(request=request)


def model(model_name: str) -> Middleware:
    """Set the name of the model to use. Typically this is automatically set inside connect."""
    return ModelMiddleware(model_name)


@dataclass(frozen=True)
class OptionsMiddleware(Middleware):
    options: dict

    def invoke(self, request: Request, next: LanguageModel) -> Reply:
        contexture = request.contexture.model_copy(
            update=dict(options=request.contexture.options | self.options)
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


@dataclass(frozen=True)
class FormatMiddleware(Middleware):
    """Request the reply used as specific format.

    If schema is None, then(unrestricted) JSON is returned
    If schema is (an instance of) a dict, then this is the schema, and JSON is returned
    If schema is a type, then we generate a schema for this type, and JSON is returned

    Extensions:
      - We support list[<type>], because list[str] is quite useful.
      - We support <type> | None as an optional type


    """

    schema: None | dict | Type

    def invoke(self, request: Request, next: LanguageModel):
        schema = self.schema
        if schema is None:
            format = "json"
        elif isinstance(schema, dict):
            format = schema
        elif isinstance(schema, (type, types.GenericAlias)) or get_origin(schema) in {
            types.UnionType,
            Union,
            list,
            tuple,
        }:
            format = jsonref.replace_refs(
                TypeAdapter(schema).json_schema(), proxies=False
            )
        else:
            assert False, f"unsupported schema: {schema}:{type(schema)}"

        request = request.model_copy(update=dict(format=format))
        reply = next.ask(request=request)

        reply = reply.reify()
        if (
            isinstance(schema, type)
            and not isinstance(schema, types.GenericAlias)
            and issubclass(schema, BaseModel)
        ):
            return reply.map(lambda content: schema.model_validate_json(content))

        return reply.map(lambda content: json.loads(content))


def format(schema: None | dict | Type = None) -> Middleware:
    """Request the output in JSON, or parsed JSON.

    If schema is None, then (unrestricted) JSON is returned
    If schema is (an instance of) a dict, then this is the schema, and JSON is returned
    If schema is a type, then we generate a schema for this type, and JSON is returned
    """
    return FormatMiddleware(schema)


def dedent() -> Middleware:
    return EmptyMiddleware()


@dataclass(frozen=True)
class ToolMiddleware(Middleware):
    """internal middleware for plumbing tools schemas"""

    tool_schemas: tuple[dict, ...]

    def invoke(self, request: Request, next: LanguageModel) -> Reply:
        request = request.model_copy(
            update=dict(tools=request.tools + self.tool_schemas)
        )
        return next.ask(request=request)

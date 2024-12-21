from __future__ import annotations
import json
import logging
import sqlite3
import textwrap
from abc import ABC
from dataclasses import asdict, dataclass, field, fields
from itertools import tee
from typing import AnyStr, Callable, Optional, Self, Tuple, NoReturn
from pydantic import BaseModel

from yaspin import yaspin
from frozendict import frozendict

from .exceptions import *
from .languagemodel import *
from .middleware import *
from .ollama import Ollama
from .cache import *
from .render import render_system, render_interaction

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Settings:
    """Local settings."""

    cache: str = None

    outdent: bool = True

    service: "Middleware" = None

    middleware: Middleware = field(default_factory=EmptyMiddleware)

    def copy(self, **update):
        return Settings(
            **{
                f.name: update[f.name] if f.name in update else getattr(self, f.name)
                for f in fields(self)
            }
        )


@dataclass(frozen=True)
class Service(ABC):
    service: "ServiceProvider"

    def list(self):
        return self.service.list()

    def __or__(self, other: Middleware):
        assert isinstance(other, Middleware), "Can only pipe with middleware"
        return Model(
            settings=Settings(service=self.service, middleware=other),
            contexture=Contexture(),
        )


@dataclass(frozen=True)
class Model(ABC):

    #    configuration: Configuration
    settings: Settings
    contexture: Contexture

    def chat(
        self,
        prompt: str,
        format: str | dict = "",
        images: list[AnyStr] = [],
        middleware: Middleware | None = None,
        raw: bool = False,
    ) -> Response:
        """
        Take a prompt and call the LLM in a previously provided context.

        Args:
            prompt (str): the prompt
            format (str | dict): the requested format of the reply
            images: (list): images to pass to the LLM
            middleware (Middleware): extra middleware specifically for this prompt
            fresh (bool): ignore any previously cached results
            raw (bool): do not reformat the prompt (rarely needed)

        Returns:
            A Response that contains the reply, and context for any future
            calls to chat.
        """
        if not raw:
            prompt = textwrap.dedent(prompt).strip()

        logger.info(f"chat prompt {prompt}")

        request, response = self.ask(prompt, format, images, middleware)

        return self.process(request, response)

    def ask(
        self,
        prompt: str,
        format: str | dict = "",
        images: list[AnyStr] = [],
        middleware: Middleware | None = None,
    ) -> tuple[Request, Reply]:
        """
        Take a prompt and call the LLM in a previously provided context.

        Args:
            prompt (str): the prompt
            format (str | dict): the requested format of the reply
            images: (list): images to pass to the LLM
            middleware (Middleware): extra middleware specifically for this prompt

        Returns:
            An internal Request/Response pair.
        """
        assert prompt is not None, "Can not build a response with no prompt"

        request = self.request(prompt, images=images, format=format)

        if middleware is not None:
            middleware = self.settings.middleware | middleware
        else:
            middleware = self.settings.middleware

        response = middleware.invoke(request=request, next=self.settings.service)

        return (request, response)

    def process(self, request: Request, response: Reply) -> "Response":

        return self.response(
            request.prompt,
            str(response),
            images=tuple(request.images),
            metrics=response.metrics(),
        )

    def request(
        self,
        prompt: str | None,
        images: list[AnyStr] = [],
        format: str | dict = "",
        fresh: bool = False,
    ) -> Request:
        return Request(
            contexture=self.contexture,
            prompt=prompt,
            images=tuple(images),
            format=format,
            fresh=fresh,
            stream=False,
        )

    def response(
        self,
        prompt: str,
        reply: str,
        images: list[AnyStr] = [],
        metrics: Metrics | None = None,
    ):
        assert isinstance(prompt, str)
        assert isinstance(reply, str)
        assert isinstance(metrics, (Metrics, type(None)))
        return Response(
            settings=self.settings,
            contexture=self.contexture.append_exchange(
                Exchange(prompt=prompt, images=tuple(images), reply=reply)
            ),
            parent=self,
            metrics=metrics,
        )

    def children(self, prompt: str | None = None, images: list[str] | None = []):
        """Return all already cached replies to this prompt."""

        service = self.settings.service
        first = self.settings.middleware.first()

        if not isinstance(first, CacheMiddleware):
            # only top-level cache can be interrogated.
            raise LLMInternalError(
                ".children(...) method needs cache to be final middleware"
            )

        replies = first.children(self.request(prompt, images=images))

        return [
            self.response(prompt_, prose, images=list(images))
            for prompt_, images, prose in replies
        ]

    def render(self) -> str:
        """Return a markdown string of the context."""
        return render_system(self.contexture.system)

    def copy(self, **update):
        return Model(
            **{
                f.name: update[f.name] if f.name in update else getattr(self, f.name)
                for f in fields(self)
            }
        )

    # Content methods

    def system(self, prompt: str) -> Self:
        """provide a system prompt."""
        return self.copy(
            contexture=self.contexture.model_copy(update=dict(system=prompt))
        )

    def outdent(self, outdent: bool = True) -> Self:
        return self.copy(settings=self.settings.copy(outdent=outdent))

    def load(self, markdown: str, complete: bool = False) -> Self:
        """Read markdown as system + prompt-reply pairs."""

        lines = markdown.split("\n")
        result = []
        if not lines:
            return self

        current_block = []
        current_is_quote = lines[0].startswith("> ")
        starts_with_quote = current_is_quote

        for line in lines:
            is_quote = line.startswith("> ")
            # Remove the '> ' prefix if it's a quote line
            line_content = line[2:] if is_quote else line
            if is_quote == current_is_quote:
                current_block.append(line_content)
            else:
                result.append("\n".join(current_block))
                current_block = [line_content]
                current_is_quote = is_quote

        # Append the last block
        if current_block:
            result.append("\n".join(current_block))

        model = self

        if not starts_with_quote:
            if sys_prompt := result[0].strip():
                # only use non-empty system prompts
                model = model.system(sys_prompt)
            result = result[1:]

        while result:
            prompt = result[0].strip()
            if len(result) == 1:
                reply = ""
            else:
                reply = result[1].strip()

            if complete and reply in ("", "..."):
                model = model.chat(prompt)
            else:
                model = model.response(prompt, reply)

            result = result[2:]

        return model

    def options(
        self,
        **kwargs,
    ):
        """Options to pass to ollama, such as temperature and seed.

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
        return self.copy(contexture=self.contexture.add_options(**kwargs))

    def middleware(self, after: Middleware) -> Self:
        """helper method to append middleware to a model

        Same as `self | after`.
        Sometimes it is cleaner in code to use the method.
        """
        return self | after

    def __or__(self, other: Middleware) -> Self:
        """pipe to append middleware to a model"""
        assert isinstance(other, Middleware), "Can only pipe with middleware"
        return self.copy(
            settings=self.settings.copy(middleware=self.settings.middleware | other)
        )


@dataclass(frozen=True)
class Response(Model):

    parent: Model
    metrics: Metrics | None

    @property
    def prompt(self) -> str:
        assert len(self.contexture.context) > 0
        return self.contexture.context[-1].prompt

    @property
    def reply(self) -> str:
        assert len(self.contexture.context) > 0
        return self.contexture.context[-1].reply

    @property
    def value(self) -> dict:
        """return a value of the reply in its requested format"""
        try:
            return json.loads(self.reply)
        except json.JSONDecodeError as e:
            return None

    def parse(self, cls: Type[BaseModel]) -> BaseModel:
        return cls.model_validate_json(str(self))

    def render(self) -> str:
        """Return a markdown string of the context."""
        return render_interaction(self.parent.render(), self.prompt, self.reply)

    def copy(self, **update):
        return Response(
            **{
                f.name: update[f.name] if f.name in update else getattr(self, f.name)
                for f in fields(self)
            }
        )

    def __str__(self):
        return self.reply

    def reject(self, message: str | None = None) -> NoReturn:
        if message:
            raise LLMResultError(message)
        raise LLMResultError()


def valid_json(response):
    """Check to see if the response reply is valid JSON."""
    return response.value is not None


def accept(response):
    """Ask the user if a response was acceptable."""
    answer = input("Accept? (Y/n)")
    return answer != "n"


class Services:
    """Internal class with lazy instantiation of services"""

    def __init__(self) -> None:
        self._model_providers = {}
        self._cache = {}

    def cache(self, filename):
        if filename not in self._cache:
            self._cache[filename] = Cache(filename)
        return self._cache[filename]


services = Services()


def connect(
    modelname: str | None = None,
    hostname: str | None = None,
    service: ServiceProvider | None = None,
) -> Model | Service:
    """return a model that uses the given model name."""

    assert (
        hostname is None or service is None
    ), "can not use hostname with a provided service provider (pass hostname directly)"

    if service is None:
        service = Ollama(hostname)

    service = Service(service)

    if modelname is not None:
        return service | model(modelname)

    return service

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field, replace
from typing import Self

from pydantic import BaseModel

from .types import (
    ServiceProvider,
    Metrics,
    Contexture,
    Request,
    Reply,
    Exchange,
    EmptyMiddleware,
)
from .exceptions import LLMInternalError
from .middleware import Middleware, CacheMiddleware
from .render import render_interaction, render_system


@dataclass(frozen=True)
class Settings:
    """Local settings."""

    service: Middleware = None

    middleware: Middleware = field(default_factory=EmptyMiddleware)


@dataclass(frozen=True)
class Service(ABC):
    service: ServiceProvider

    def list(self):
        return self.service.list()

    def __or__(self, other: Middleware) -> Model:
        assert isinstance(other, Middleware), "Can only pipe with middleware"
        return Model(
            settings=Settings(service=self.service, middleware=other),
            contexture=Contexture(),
        )


@dataclass(frozen=True)
class Model(ABC):

    settings: Settings
    contexture: Contexture

    def chat(
        self,
        prompt: str,
        images: list[str] = [],
        middleware: Middleware | None = None,
    ) -> Response:
        """
        Take a prompt and call the LLM in a previously provided context.

        Args:
            prompt (str): the prompt
            images: (list): images to pass to the LLM
            middleware (Middleware): extra middleware specifically for this prompt

        Returns:
            A Response that contains the reply, and context for any future
            calls to chat.
        """
        # if not raw:
        #     prompt = textwrap.dedent(prompt).strip()

        request, response = self.ask(prompt, images, middleware)

        return self.process(request, response)

    def ask(
        self,
        prompt: str,
        images: list[str] = [],
        middleware: Middleware | None = None,
    ) -> tuple[Request, Reply]:
        """
        Take a prompt and call the LLM in a previously provided context.

        Args:
            prompt (str): the prompt
            images: (list): images to pass to the LLM
            middleware (Middleware): extra middleware specifically for this prompt

        Returns:
            An internal Request/Response pair.
        """
        assert prompt is not None, "Can not build a response with no prompt"

        request = self.request(prompt, images=images)

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
            value=response.value,
        )

    def request(
        self,
        prompt: str | None,
        images: list[str] = [],
        format: str | dict = "",
        fresh: bool = False,
        stream: bool = False,
    ) -> Request:
        return Request(
            contexture=self.contexture,
            prompt=prompt,
            images=tuple(images),
            format=format,
            fresh=fresh,
            stream=stream,
        )

    def response(
        self,
        prompt: str,
        reply: str,
        images: list[str] = [],
        metrics: Metrics | None = None,
        value: BaseModel | dict | None = None,
    ):
        assert isinstance(prompt, str)
        assert isinstance(reply, str)
        assert isinstance(metrics, (Metrics, type(None)))
        assert isinstance(value, (BaseModel, dict, type(None)))
        return Response(
            settings=self.settings,
            contexture=self.contexture.append_exchange(
                Exchange(prompt=prompt, images=tuple(images), reply=reply)
            ),
            parent=self,
            metrics=metrics,
            value=value,
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

    # Content methods

    def system(self, prompt: str) -> Self:
        """provide a system prompt."""
        return replace(
            self, contexture=self.contexture.model_copy(update=dict(system=prompt))
        )

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

    def __or__(self, other: Middleware) -> Self:
        """pipe to append middleware to a model"""
        assert isinstance(other, Middleware), "Can only pipe with middleware"
        return replace(
            self,
            settings=replace(
                self.settings, middleware=self.settings.middleware | other
            ),
        )


@dataclass(frozen=True)
class Response(Model):

    parent: Model
    metrics: Metrics | None
    value: BaseModel | dict | None

    @property
    def prompt(self) -> str:
        assert len(self.contexture.context) > 0
        return self.contexture.context[-1].prompt

    @property
    def reply(self) -> str:
        assert len(self.contexture.context) > 0
        return self.contexture.context[-1].reply

    def render(self) -> str:
        """Return a markdown string of the context."""
        return render_interaction(self.parent.render(), self.prompt, self.reply)

    def __str__(self):
        return self.reply

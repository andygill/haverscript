from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field, replace
from typing import Any
import json
from pydantic import BaseModel, ConfigDict

from .types import (
    ServiceProvider,
    Metrics,
    Contexture,
    Request,
    Reply,
    Exchange,
    Prompt,
    RequestMessage,
    EmptyMiddleware,
    AssistantMessage,
    ToolResult,
    ToolReply,
    ToolCall,
)
from .exceptions import LLMInternalError
from .middleware import Middleware, CacheMiddleware
from .render import render_interaction, render_system
from .markdown import Markdown
from .tools import Tools


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


class Model(BaseModel):

    settings: Settings
    contexture: Contexture

    model_config = ConfigDict(frozen=True)

    def chat(
        self,
        prompt: str | Markdown,
        images: list[str] = [],
        middleware: Middleware | None = None,
        tools: Tools | None = None,
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
        if isinstance(prompt, Markdown):
            prompt = str(prompt)

        request, reply = self.ask(
            Prompt(content=prompt, images=images), middleware, tools
        )

        response = self.process(request, reply)

        while response.tool_calls:
            assert (
                tools is not None
            ), "internal error: tools should be provided to handle tool calls"
            results = []
            for tool in response.tool_calls:
                output = tools(tool.name, tool.arguments)
                results.append(
                    ToolReply(id=tool.id, name=tool.name, content=str(output))
                )

            request, reply = response.ask(
                ToolResult(results=results), middleware, tools
            )
            response = response.process(request, reply)

        return response

    def ask(
        self,
        prompt: RequestMessage,
        middleware: Middleware | None = None,
        tools: Tools | None = None,
    ) -> tuple[Request, Reply]:
        """
        Take a prompt and call the LLM in a previously provided context.

        Args:
            prompt (RequestMessage): the prompt in message format
            images: (list): images to pass to the LLM
            middleware (Middleware): extra middleware specifically for this prompt

        Returns:
            An internal Request/Response pair.
        """
        assert prompt is not None, "Can not build a response with no prompt"

        request = self.request(prompt, tools=tools)

        if middleware is not None:
            middleware = self.settings.middleware | middleware
        else:
            middleware = self.settings.middleware

        response = middleware.invoke(request=request, next=self.settings.service)

        return (request, response)

    def process(self, request: Request, response: Reply) -> Response:

        return self.response(
            request.prompt,
            str(response),
            metrics=response.metrics(),
            value=response.value,
            tool_calls=tuple(response.tool_calls()),
        )

    def request(
        self,
        prompt: RequestMessage | None,
        format: str | dict = "",
        fresh: bool = False,
        stream: bool = False,
        tools: Tools | None = None,
    ) -> Request:

        tool_schemas = tools.tool_schemas() if tools else ()

        return Request(
            contexture=self.contexture,
            prompt=prompt,
            format=format,
            fresh=fresh,
            stream=stream,
            tools=tool_schemas,
        )

    def response(
        self,
        prompt: RequestMessage,
        reply: str,
        metrics: Metrics | None = None,
        value: Any | None = None,
        tool_calls: tuple[ToolCall, ...] = (),
    ):
        assert isinstance(prompt, RequestMessage)
        assert isinstance(metrics, (Metrics, type(None)))
        assert isinstance(tool_calls, tuple) and all(
            isinstance(tc, ToolCall) for tc in tool_calls
        )
        return Response(
            settings=self.settings,
            contexture=self.contexture.append_exchange(
                Exchange(
                    prompt=prompt,
                    reply=AssistantMessage(content=reply),
                )
            ),
            parent=self,
            metrics=metrics,
            value=value,
            tool_calls=tool_calls,
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

        if prompt is not None:
            prompt = Prompt(content=prompt, images=tuple(images))

        replies = first.children(self.request(prompt))

        return [
            self.response(Prompt(content=prompt_, images=tuple(images)), prose)
            for prompt_, images, prose in replies
        ]

    def render(self) -> str:
        """Return a markdown string of the context."""
        return render_system(self.contexture.system)

    # Content methods

    def system(self, prompt: str | Markdown) -> Model:
        """provide a system prompt."""
        if isinstance(prompt, Markdown):
            prompt = str(prompt)

        return self.model_copy(
            update=dict(
                contexture=self.contexture.model_copy(update=dict(system=prompt))
            )
        )

    def load(self, markdown: str, complete: bool = False) -> Model:
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
                model = model.response(Prompt(content=prompt), reply)

            result = result[2:]

        return model

    def __or__(self, other: Middleware) -> Model:
        """pipe to append middleware to a model"""
        assert isinstance(other, Middleware), "Can only pipe with middleware"
        return self.model_copy(
            update=dict(
                settings=replace(
                    self.settings, middleware=self.settings.middleware | other
                )
            )
        )


class Response(Model):

    parent: Model
    metrics: Metrics | None
    value: Any | None
    tool_calls: tuple[ToolCall, ...]

    @property
    def prompt(self) -> str:
        assert len(self.contexture.context) > 0
        return self.contexture.context[-1].prompt.content

    @property
    def reply(self) -> str:
        assert len(self.contexture.context) > 0
        return self.contexture.context[-1].reply.content

    def render(self) -> str:
        """Return a markdown string of the context."""
        return render_interaction(self.parent.render(), self.contexture.context[-1])

    def __str__(self):
        return self.reply

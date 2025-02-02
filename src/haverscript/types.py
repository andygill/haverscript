from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Callable, Any

from pydantic import BaseModel, ConfigDict, Field


class Extras(BaseModel):
    """Extras is a place to put extra information in a LLM response."""

    model_config = ConfigDict(frozen=True)


class Metrics(Extras):
    pass


class Informational(Extras):
    """Background information, typically from a middleware component."""

    message: str


class Value(Extras):
    value: Any


class ToolCall(Extras):
    """A tool call is a response from an LLM that requestsd calling a named tool."""

    name: str
    arguments: dict


class Message(BaseModel):
    """A message is content with a role (assistant, user, system, tool)."""

    content: str

    model_config = ConfigDict(frozen=True)

    def role_content_json(self):
        result = {}
        result["role"] = self.role
        result["content"] = self.content
        if hasattr(self, "images") and self.images:
            result["images"] = self.images
        return result


class SystemMessage(Message):
    role: str = "system"


class RequestMessage(Message):
    pass


class Prompt(RequestMessage):
    role: str = "user"
    images: tuple[str, ...] = ()


class ToolResult(RequestMessage):
    role: str = "tool"


class ResponseMessage(Message):
    pass


class AssistantMessage(ResponseMessage):
    role: str = "assistant"


class Exchange(BaseModel):
    prompt: RequestMessage
    reply: ResponseMessage

    model_config = ConfigDict(frozen=True)


class Contexture(BaseModel):
    """Background parts of a request"""

    context: tuple[Exchange, ...] = ()
    system: str | None = None
    options: dict = Field(default_factory=dict)
    model: str | None = None

    model_config = ConfigDict(frozen=True)

    def append_exchange(self, exchange: Exchange):
        return self.model_copy(update=dict(context=self.context + (exchange,)))

    def add_options(self, **options):
        # using this pattern exclude None value in dict
        return self.model_copy(
            update=dict(
                options=dict(
                    {
                        key: value
                        for key, value in {**self.options, **options}.items()
                        if value is not None
                    }
                )
            )
        )


class Request(BaseModel):
    """Foreground parts of a request"""

    contexture: Contexture
    prompt: Prompt | None

    stream: bool = False
    fresh: bool = False

    # images: tuple[str, ...] = ()
    format: str | dict = ""  # str is "json" or "", dict is a JSON schema
    tools: tuple[dict] = ()

    model_config = ConfigDict(frozen=True)


class Reply:
    """A potentially tokenized response to a large language model"""

    def __init__(self, packets: Iterable[str | Extras]):
        self._packets = iter(packets)
        # We always have at least one item in our sequence.
        # This typically will cause as small pause before
        # returning the Reply constructor.
        # It does mean, by design, that the generator
        # has started producing tokens, so the context
        # has already been processed. If you have a
        # Reply, you can assume that tokens
        # are in flight, and the LLM worked.
        try:
            self._cache = [next(self._packets)]
        except StopIteration:
            self._packets = iter([])
            self._cache = []
        self._lock = threading.Lock()
        self.closers = []
        self.closing = False

    def __str__(self):
        return "".join(self.tokens())

    def __repr__(self):

        return f"Reply([{', '.join([repr(t) for t in self._cache])}{']' if self.closing else ', ...'})"

    def __iter__(self):
        ix = 0
        while True:
            # We need a lock, because the contents can be consumed
            # by difference threads. With list, this just works.
            # With generators, we need to both the cache, and the lock.
            with self._lock:
                if ix < len(self._cache):
                    result = self._cache[ix]
                else:
                    assert ix == len(self._cache)
                    try:
                        result = next(self._packets)
                    except StopIteration:
                        break
                    self._cache.append(result)

            ix += 1  # this a local ix, so does not need guarded
            yield result

        # auto close
        with self._lock:
            if self.closing:
                return
            # first past the post
            self.closing = True

        # close all completers
        for completion in self.closers:
            completion()

    def tokens(self) -> Iterable[str]:
        """Returns all str tokens."""
        yield from (token for token in self if isinstance(token, str))

    def metrics(self) -> Metrics | None:
        """Returns any Metrics."""
        for t in self:
            if isinstance(t, Metrics):
                return t
        return None

    @property
    def value(self) -> dict | BaseModel | None:
        """Returns any value build by format middleware.

        value is a property to be consistent with Response.
        """
        for t in self:
            if isinstance(t, Value):
                return t.value
        return None

    def after(self, completion: Callable[[], None]) -> None:
        with self._lock:
            if not self.closing:
                self.closers.append(completion)
                return

        # we have completed, so just call completion callback.
        completion()

    def __add__(self, other: "Reply"):

        # Need to append both streams of tokens and other values.
        # Need to correctly thread the close,
        # because the outer close needs the inner close to be called.
        def streaming():
            yield from self
            yield from other

        return Reply(streaming())


class LanguageModel(ABC):
    """Base class for anything that can by asked things, that is takes a configuration/prompt and returns token(s)."""

    @abstractmethod
    def ask(self, request: Request) -> Reply:
        """Ask a LLM a specific request."""

    def __or__(self, other) -> LanguageModel:
        assert isinstance(other, Middleware)
        return MiddlewareLanguageModel(other, self)


class ServiceProvider(LanguageModel):
    """A ServiceProvider is a LanguageModel that serves specific models."""

    @abstractmethod
    def list(self) -> list[str]:
        """Return the list of valid models for this provider."""


@dataclass(frozen=True)
class Middleware(ABC):
    """Middleware is a bidirectional Prompt and Reply processor.

    Middleware is something you use on a LanguageModel,
    and a LanguageModel is something you *call*.
    """

    @abstractmethod
    def invoke(self, request: Request, next: LanguageModel) -> Reply:
        return next.ask(request=request)

    def first(self):
        """get the first Middleware in the pipeline (from the Prompt's point of view)"""
        return self

    def __or__(self, other: LanguageModel) -> Middleware:
        return AppendMiddleware(self, other)


@dataclass(frozen=True)
class MiddlewareLanguageModel(LanguageModel):
    """MiddlewareLanguageModel is Middleware with a specific target LanguageModel.

    This combination of Middleware and LanguageModel is itself a LanguageModel.
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

    def first(self):
        if first := self.before.first():
            return first
        return self.after.first()


@dataclass(frozen=True)
class EmptyMiddleware(Middleware):

    def invoke(self, request: Request, next: LanguageModel) -> Reply:
        return next.ask(request=request)

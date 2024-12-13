from abc import ABC, abstractmethod
from dataclasses import dataclass
from concurrent.futures import Future
import threading
from collections.abc import Iterable
from typing import Iterator
from typing import Callable
from pydantic import BaseModel, ConfigDict, Field
from frozendict import frozendict


@dataclass(frozen=True)
class Metrics(ABC):
    pass


class LanguageModelExchange(BaseModel):
    prompt: str
    images: tuple[str, ...] | None
    reply: str

    model_config = ConfigDict(frozen=True)


class LanguageModelContexture(BaseModel):
    """Background parts of a request"""

    context: tuple[LanguageModelExchange, ...] = ()
    system: str | None = None
    options: dict = Field(default_factory=dict)
    model: str

    # TODO: to move both of these into LanguageModelRequest
    images: tuple[str, ...] = ()
    format: str | dict = ""  # str is "json" or "", dict is a JSON schema

    model_config = ConfigDict(frozen=True)

    def append_exchange(self, exchange: LanguageModelExchange):
        return self.model_copy(
            update={"context": self.context + (exchange,), "images": ()}
        )

    def append_image(self, image: str):
        return self.model_copy(update={"images": self.images + (image,)})

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


class LanguageModelRequest(BaseModel):
    """Foreground parts of a request"""

    contexture: LanguageModelContexture
    prompt: str
    stream: bool

    model_config = ConfigDict(frozen=True)


class LanguageModelResponse:
    """A potentially tokenized response to a large language model"""

    def __init__(self, packets: Iterable[str | Metrics]):
        self._packets = iter(packets)
        # We always have at least one item in our sequence.
        # This typically will cause as small pause before
        # returning the LanguageModelResponse constructor.
        # It does mean, by design, that the generator
        # has started producing tokens, so the context
        # has already been processed. If you have a
        # LanguageModelResponse, you can assume that tokens
        # are in flight, and the LLM worked.
        try:
            self._cache = [next(self._packets)]
        except StopIteration:
            self._packets = iter([])
            self._cache = []
        self._lock = threading.Lock()
        self.closers = []

    def __str__(self):
        return "".join(self.tokens())

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
                        return
                    self._cache.append(result)
            ix += 1
            yield result

    def tokens(self) -> Iterable[str]:
        """Returns all str tokens."""
        yield from (token for token in self if isinstance(token, str))

    def metrics(self) -> Metrics | None:
        """Returns any Metrics."""
        for t in self:
            if isinstance(t, Metrics):
                return t
        return None

    def after(self, completion: Callable[[], None]) -> None:
        self.closers.append(completion)

    def close(self):
        for completion in self.closers:
            completion()


class LanguageModel(ABC):
    """Base class for anything that chats, that is takes a configuration and prompt and returns token(s)."""

    @abstractmethod
    def chat(self, prompt: str, *kwargs) -> LanguageModelResponse:
        """Call the chat method of an LLM.

        prompt is the main text
        ksargs contains a dictionary of configuration options

        returns a LanguageModelResponse.
        """
        pass


class ServiceProvider(LanguageModel):
    """A ServiceProvider is a LanguageModel that serves specific models."""

    def list(self) -> list[str]:
        models = self.client[self.hostname].list()
        assert "models" in models
        return [model["name"] for model in models["models"]]

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Metrics(ABC):
    pass


class LanguageModelResponse:
    """A tokenized response to a large language model"""

    def __init__(self, tokens):
        self.tokens = iter(tokens)
        self.cache = []

    def __iter__(self):
        return self._ResponseIterator(self.cache, self.tokens)

    def __str__(self):
        return "".join([t for t in self if isinstance(t, str)])

    def metrics(self) -> Metrics | None:
        for t in self:
            if isinstance(t, Metrics):
                return t
        return None

    class _ResponseIterator:
        def __init__(self, cache, iterable):
            self._iterable = iterable
            self._cache = cache
            self._index = 0  # Track the position in the cache

        def __iter__(self):
            return self

        def __next__(self):
            if self._index < len(self._cache):
                # Return cached value if available
                value = self._cache[self._index]
            else:
                # Otherwise, fetch the next value from the original iterator
                try:
                    value = next(self._iterable)
                    self._cache.append(value)  # Add to cache
                except StopIteration:
                    raise StopIteration
            self._index += 1
            return value


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

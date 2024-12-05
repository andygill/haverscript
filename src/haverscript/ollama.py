import copy
from dataclasses import asdict, dataclass, field, fields
from types import GeneratorType
from frozendict import frozendict
from typing import AnyStr, Callable, Optional, Self, Tuple, NoReturn, Generator

import ollama

from .languagemodel import Metrics, ServiceProvider, LanguageModelResponse


@dataclass(frozen=True)
class Configuration:
    """This will be factored out soon"""

    options: frozendict = field(default_factory=frozendict)
    json: bool = False
    system: Optional[str] = None
    context: Tuple[  # list (using a tuple) of prompt+images response triples
        Tuple[str, Tuple[str, ...], str], ...
    ] = ()
    images: Tuple[str, ...] = ()  # tuple of images


@dataclass(frozen=True)
class OllamaMetrics(Metrics):
    total_duration: int  # time spent generating the response
    load_duration: int  # time spent in nanoseconds loading the model
    prompt_eval_count: int  # number of tokens in the prompt
    prompt_eval_duration: int  # time spent in nanoseconds evaluating the prompt
    eval_count: int  # number of tokens in the response
    eval_duration: int  # time in nanoseconds spent generating the response


class Ollama(ServiceProvider):
    client = {}

    def __init__(self, hostname: str | None = None) -> None:
        self.hostname = hostname
        if hostname not in Ollama.client:
            Ollama.client[hostname] = ollama.Client(host=hostname)

    def list(self) -> list[str]:
        models = self.client[self.hostname].list()
        assert "models" in models
        return [model["name"] for model in models["models"]]

    def _suggestions(self, e: Exception):
        # Slighty better message. Should really have a type of reply for failure.
        if "ConnectError" in str(type(e)):
            print("Connection error (Check if ollama is running)")
        return e

    def generator(self, response):

        if isinstance(response, GeneratorType):
            try:
                for chunk in response:
                    if chunk["done"]:
                        yield OllamaMetrics(
                            **{
                                k: chunk[k]
                                for k in OllamaMetrics.__dataclass_fields__.keys()
                            }
                        )
                    yield from chunk["message"]["content"]
            except Exception as e:
                raise self._suggestions(e)
        else:
            assert isinstance(response["message"]["content"], str)
            yield response["message"]["content"]
            yield OllamaMetrics(
                **{k: response[k] for k in OllamaMetrics.__dataclass_fields__.keys()}
            )

    def chat(self, prompt: str, model: str, **kwargs):

        configuration = Configuration(
            options=kwargs["options"],
            json=kwargs["json"],
            system=kwargs["system"],
            context=kwargs["context"],
            images=kwargs["images"],
        )

        messages = []

        stream = "stream" in kwargs and kwargs["stream"]

        options = copy.deepcopy(kwargs["options"]) if "options" in kwargs else {}

        if "system" in kwargs and kwargs["system"]:
            messages.append({"role": "system", "content": configuration.system})

        for pmt, imgs, resp in configuration.context:
            messages.append(
                {"role": "user", "content": pmt}
                | ({"images": list(imgs)} if imgs else {})
            )
            messages.append({"role": "assistant", "content": resp})

        messages.append(
            {"role": "user", "content": prompt}
            | ({"images": list(configuration.images)} if configuration.images else {})
        )

        try:
            response = self.client[self.hostname].chat(
                model=model,
                stream=stream,
                messages=messages,
                options=options,
                format="json" if configuration.json else "",
            )

            return LanguageModelResponse(self.generator(response))

        except Exception as e:
            raise self._suggestions(e)

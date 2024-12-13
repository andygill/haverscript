import copy
from dataclasses import asdict, dataclass, field, fields
from types import GeneratorType
from frozendict import frozendict
from typing import AnyStr, Callable, Optional, Self, Tuple, NoReturn, Generator

import ollama

from .languagemodel import (
    Metrics,
    ServiceProvider,
    LanguageModelResponse,
    LanguageModelRequest,
)


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

    def chat(self, request: LanguageModelRequest):

        prompt = request.prompt
        model = request.contexture.model

        kwargs = {}
        kwargs["options"] = request.contexture.options
        kwargs["system"] = request.contexture.system
        kwargs["context"] = request.contexture.context
        kwargs["images"] = request.images
        kwargs["stream"] = request.stream

        messages = []

        stream = "stream" in kwargs and kwargs["stream"]

        options = copy.deepcopy(kwargs["options"]) if "options" in kwargs else {}

        if request.contexture.system:
            messages.append({"role": "system", "content": request.contexture.system})

        for exchange in request.contexture.context:
            pmt, imgs, resp = exchange.prompt, exchange.images, exchange.reply
            messages.append(
                {"role": "user", "content": pmt}
                | ({"images": list(imgs)} if imgs else {})
            )
            messages.append({"role": "assistant", "content": resp})

        messages.append(
            {"role": "user", "content": prompt}
            | ({"images": list(request.images)} if request.images else {})
        )

        try:
            response = self.client[self.hostname].chat(
                model=model,
                stream=stream,
                messages=messages,
                options=options,
                format=request.format,
            )

            return LanguageModelResponse(self.generator(response))

        except Exception as e:
            raise self._suggestions(e)

import json
import logging
import os
from types import GeneratorType

from dataclasses import dataclass

import together
import requests


from .haverscript import Metrics, Model, Service, model
from .languagemodel import ServiceProvider, Reply, Request
from .exceptions import LLMRateLimitError


@dataclass(frozen=True)
class TogetherMetrics(Metrics):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Together(ServiceProvider):
    client: together.Together | None = None
    hostname = ""

    def __init__(self) -> None:
        key = os.getenv("TOGETHER_API_KEY")
        assert key is not None, "TOGETHER_API_KEY is not set to key"
        if self.client is None:
            self.client: Together = together.Together(api_key=key)

    def list(self) -> list[str]:
        models = self.client.models.list()
        return [model.id for model in models]

    def _suggestions(self, e: Exception):
        # Slighty better message. Should really have a type of reply for failure.
        if "ConnectError" in str(type(e)):
            print("Connection error with together.ai")
        return e

    def metrics(self, chunk: dict) -> Metrics:
        return TogetherMetrics(
            **{k: chunk[k] for k in TogetherMetrics.__dataclass_fields__.keys()}
        )

    def generator(self, response):

        if isinstance(response, GeneratorType):
            try:
                for chunk in response:
                    if chunk.choices[0].finish_reason:
                        yield self.metrics(chunk.usage.dict())
                    yield from chunk.choices[0].delta.content
            except Exception as e:
                raise self._suggestions(e)
        else:
            assert isinstance(response.choices[0].message.content, str)
            yield response.choices[0].message.content
            yield self.metrics(response.usage.dict())

    def ask(self, request: Request):

        messages = []

        if request.contexture.system:
            messages.append({"role": "system", "content": request.contexture.system})

        for exchange in request.contexture.context:
            assert not exchange.images, "images not (yet) supported"
            messages.append({"role": "user", "content": exchange.prompt})
            messages.append({"role": "assistant", "content": exchange.reply})

        assert not request.images, "images not (yet) supported"
        messages.append({"role": "user", "content": request.prompt})

        try:
            assert isinstance(self.client, together.Together)
            response = self.client.chat.completions.create(
                model=request.contexture.model,
                stream=request.stream,
                messages=messages,
                options=request.contexture.options,
                format=request.format,
            )

            return Reply(self.generator(response))

        except Exception as e:
            raise self._suggestions(e)


def connect(model_name: str | None = None) -> Model | Service:
    service = Service(Together())
    if model_name:
        service = service | model(model_name)
    return service

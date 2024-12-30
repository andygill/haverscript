from dataclasses import dataclass
from types import GeneratorType

import ollama

from .haverscript import Model, Service
from .types import (
    Metrics,
    ServiceProvider,
    Reply,
    Request,
)
from .middleware import model


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
        if hostname not in self.client:
            self.client[hostname] = ollama.Client(host=hostname)

    def list(self) -> list[str]:
        models = self.client[self.hostname].list()
        assert "models" in models
        return [model.model for model in models["models"]]

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
                    yield chunk["message"]["content"]
            except Exception as e:
                raise self._suggestions(e)
        else:
            assert isinstance(response["message"]["content"], str)
            yield response["message"]["content"]
            yield OllamaMetrics(
                **{k: response[k] for k in OllamaMetrics.__dataclass_fields__.keys()}
            )

    def ask(self, request: Request):

        messages = []

        if request.contexture.system:
            messages.append({"role": "system", "content": request.contexture.system})

        for exchange in request.contexture.context:
            messages.append(
                {"role": "user", "content": exchange.prompt}
                | ({"images": list(exchange.images)} if exchange.images else {})
            )
            messages.append({"role": "assistant", "content": exchange.reply})

        messages.append(
            {"role": "user", "content": request.prompt}
            | ({"images": list(request.images)} if request.images else {})
        )

        try:
            response = self.client[self.hostname].chat(
                model=request.contexture.model,
                stream=request.stream,
                messages=messages,
                options=request.contexture.options,
                format=request.format,
            )

            return Reply(self.generator(response))

        except Exception as e:
            raise self._suggestions(e)


def connect(
    model_name: str | None = None,
    hostname: str | None = None,
) -> Model | Service:
    """return a model or service that uses the given model name."""

    service = Service(Ollama(hostname=hostname))

    if model_name:
        service = service | model(model_name)

    return service

from dataclasses import dataclass
from types import GeneratorType

import ollama

from .haverscript import Model, Service
from .types import Metrics, ServiceProvider, Reply, Request, SystemMessage, ToolCall
from .middleware import model


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

    def tokenize(self, chunk: dict):
        if chunk["done"]:
            yield OllamaMetrics(
                **{k: chunk[k] for k in OllamaMetrics.model_fields.keys()}
            )
        message = chunk["message"]
        if "tool_calls" in message:
            for tool in message["tool_calls"]:
                yield ToolCall(
                    name=tool.function.name,
                    arguments=tool.function.arguments,
                )
        assert isinstance(message["content"], str)
        yield message["content"]

    def generator(self, response):

        if isinstance(response, GeneratorType):
            try:
                for chunk in response:
                    yield from self.tokenize(chunk)

            except Exception as e:
                raise self._suggestions(e)

        else:
            yield from self.tokenize(response)

    def ask(self, request: Request):

        messages = []

        if request.contexture.system:
            SystemMessage(content=request.contexture.system).append_to(messages)

        for exchange in request.contexture.context:
            exchange.append_to(messages)

        request.prompt.append_to(messages)

        tools = None
        if request.tools:
            tools = list(request.tools)

        # normalize the messages for ollama
        messages = [
            {
                key: value
                for key, value in original_dict.items()
                if key in {"role", "content", "images"}
            }
            for original_dict in messages
        ]

        try:
            response = self.client[self.hostname].chat(
                model=request.contexture.model,
                stream=request.stream,
                messages=messages,
                options=request.contexture.options,
                format=request.format,
                tools=tools,
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

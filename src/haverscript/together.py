import os
from dataclasses import dataclass
import time
from types import GeneratorType
import json

import together

from .haverscript import Metrics, Model, Service
from .types import Reply, Request, ServiceProvider, SystemMessage, ToolCall
from .middleware import model


class TogetherMetrics(Metrics):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Together(ServiceProvider):
    client: together.Together | None = None
    hostname = ""

    def __init__(
        self,
        api_key: str | None = None,
        timeout: int | None = None,
        max_retries: int | None = None,
    ) -> None:
        if api_key is None:
            api_key = os.getenv("TOGETHER_API_KEY")
        assert api_key is not None, "need TOGETHER_API_KEY"
        if self.client is None:
            self.client: Together = together.Together(
                api_key=api_key, timeout=timeout, max_retries=max_retries
            )

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
            **{k: chunk[k] for k in TogetherMetrics.model_fields.keys()}
        )

    def tool_calls(self, message):
        if hasattr(message, "tool_calls") and message.tool_calls is not None:
            for tool in message.tool_calls:
                yield ToolCall(
                    name=tool.function.name,
                    arguments=json.loads(tool.function.arguments),
                    id=tool.id,
                )

    def generator(self, response):

        if isinstance(response, GeneratorType):
            try:
                for chunk in response:
                    for choice in chunk.choices:
                        if choice.finish_reason and chunk.usage:
                            yield self.metrics(chunk.usage.model_dump())
                        yield choice.delta.content
            except Exception as e:
                raise self._suggestions(e)
        else:
            for chunk in response.choices:
                if isinstance(chunk.message.content, str):
                    yield chunk.message.content
                yield from self.tool_calls(response.choices[0].message)
                yield self.metrics(response.usage.model_dump())

    def ask(self, request: Request):

        messages = []

        if request.contexture.system:
            SystemMessage(content=request.contexture.system).append_to(messages)

        for exchange in request.contexture.context:
            exchange.append_to(messages)

        request.prompt.append_to(messages)

        # normalize the messages for together
        key_map = {
            "role": "role",
            "content": "content",
            "id": "tool_call_id",
            "name": "name",
        }
        messages = [
            {
                key_map[key]: value
                for key, value in original_dict.items()
                if key in key_map
            }
            for original_dict in messages
        ]
        # remove all messages that have content == ""
        messages = [message for message in messages if message["content"] != ""]
        # if there are any role == "tool" messages, then set this to true
        tool_reply = messages[-1]["role"] == "tool"

        together_keywords = set(
            [
                "max_tokens",
                "stop",
                "temperature",
                "top_p",
                "top_k",
                "repetition_penalty",
                "presence_penalty",
                "frequency_penalty",
                "min_p",
                "logit_bias",
                "seed",
                "logprobs",
            ]
        )
        kwargs = {
            key: request.contexture.options[key]
            for key in request.contexture.options
            if key in together_keywords
        }

        response_format = None
        if format == "json":
            response_format = {"type": "json_object"}
        elif isinstance(format, dict):
            response_format = {"type": "json_object", "schema": format}

        # In together.ai, we can not use tools when re-running with a tool call response.
        if request.tools and not tool_reply:
            assert (
                request.stream is False
            ), "Can not use together function calling tools with streaming"
            kwargs["tools"] = list(request.tools)
            kwargs["tool_choice"] = "auto"

        try:
            assert isinstance(self.client, together.Together)
            response = self.client.chat.completions.create(
                model=request.contexture.model,
                stream=request.stream,
                messages=messages,
                response_format=response_format,
                **kwargs,
            )

            return Reply(self.generator(response))

        except Exception as e:
            raise self._suggestions(e)


def connect(
    model_name: str | None = None,
    api_key: str | None = None,
    timeout: int | None = None,
    max_retries: int | None = None,
) -> Model | Service:
    """return a model or service that uses the given model name."""

    service = Service(
        Together(api_key=api_key, timeout=timeout, max_retries=max_retries)
    )
    if model_name:
        service = service | model(model_name)
    return service

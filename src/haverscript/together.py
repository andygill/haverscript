import json
import logging
import os
from dataclasses import dataclass

import requests

from .haverscript import Configuration, Metrics, ServiceProvider, LanguageModelResponse
from .exceptions import LLMRateLimitError


@dataclass(frozen=True)
class TogetherMetrics(Metrics):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Together(ServiceProvider):
    def __init__(self) -> None:
        self.key = os.getenv("TOGETHER_API_KEY")
        assert self.key is not None, "TOGETHER_API_KEY is not set to key"

    def name(self):
        return f"together.xyz"

    def list(self):
        url = "https://api.together.xyz/v1/models"
        headers = {"accept": "application/json", "authorization": f"Bearer {self.key}"}
        response = requests.get(url, headers=headers)
        models = response.json()
        return [model["id"] for model in models]

    def _streaming(self, response):

        buffer = ""

        def process_packet(packet):
            if packet.startswith("data: "):
                json_str = packet[len("data: ") :]
                if json_str == "[DONE]":
                    return None
                try:
                    json_data = json.loads(json_str)
                    return json_data
                except json.JSONDecodeError as e:
                    logging.error(f"JSON decoding failed: {e}")
            else:
                packet_data = json.loads(packet)
                if (
                    "error" in packet_data
                    and "type" in packet_data["error"]
                    and packet_data["error"]["type"] == "credit_limit"
                ):
                    raise LLMRateLimitError

            return None

        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
            buffer += chunk
            while "\n\n" in buffer:
                packet, buffer = buffer.split("\n\n", 1)
                json_data = process_packet(packet.strip())
                if json_data is not None:
                    if "usage" in json_data and isinstance(json_data["usage"], dict):
                        yield TogetherMetrics(
                            **{
                                k: json_data["usage"][k]
                                for k in TogetherMetrics.__dataclass_fields__.keys()
                            }
                        )

                    yield json_data["choices"][0]["text"]

            # Handle any remaining data in the buffer after the loop ends
            if buffer.strip():
                json_data = process_packet(buffer.strip())
                if json_data is not None:
                    yield json_data["choices"][0]["text"]

    def chat(self, configuration: Configuration, prompt: str, stream: bool):
        messages = []

        if configuration.system:
            messages.append({"role": "system", "content": configuration.system})

        for pmt, imgs, resp in configuration.context:
            assert imgs == (), f"imgs={imgs}"
            messages.append({"role": "user", "content": pmt})
            messages.append({"role": "assistant", "content": resp})

        messages.append({"role": "user", "content": prompt})

        url = "https://api.together.xyz/v1/chat/completions"

        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {self.key}",
        }

        payload = {
            "model": configuration.model,
            "messages": messages,
            "stream": stream,
        } | configuration.options

        response = requests.post(url, json=payload, headers=headers, stream=stream)

        if stream:
            return LanguageModelResponse(self._streaming(response))
        else:
            reply = response.json()
            choices = reply["choices"]
            assert len(choices) == 1
            choice = choices[0]
            metrics = None
            if "usage" in reply and isinstance(reply["usage"], dict):
                metrics = TogetherMetrics(
                    **{
                        k: reply["usage"][k]
                        for k in TogetherMetrics.__dataclass_fields__.keys()
                    }
                )
            return LanguageModelResponse([choice["message"]["content"], metrics])

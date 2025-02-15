from typing import Callable, Protocol

from haverscript import *
from haverscript.types import (
    Reply,
    Prompt,
    Contexture,
    Exchange,
    AssistantMessage,
    ServiceProvider,
    Request,
)
from copy import deepcopy


class ChatBot(Protocol):
    def chat(self, text: str) -> Reply:
        """take a string, update the object, and return a Reply."""
        ...


def connect_chatbot(chatbot: ChatBot | Callable[[str], ChatBot]) -> Model:

    class ChatBotServiceProvider(ServiceProvider):
        def __init__(self):
            self._model_cache: dict[tuple[str, tuple[Exchange, ...]], ChatBot] = {}

        def list(self) -> list[str]:
            return ["chatbot"]

        def ask(self, request: Request) -> Reply:
            assert isinstance(request, Request)
            assert isinstance(request.prompt, Prompt)
            assert isinstance(request.contexture, Contexture)

            system = request.contexture.system
            context = request.contexture.context

            try:
                model = self._model_cache[system, context]
            except KeyError:
                if context == ():
                    if callable(chatbot):
                        model = chatbot(system)
                    else:
                        model = deepcopy(chatbot)
                    self._model_cache[system, ()] = model
                else:
                    # We has a context we've never seen
                    # Which means we did not generate it
                    # Which means we reject it
                    # Should never happen with regular usage
                    assert False, "unknown system or context"

            model: ChatBot = deepcopy(model)

            response: Reply = model.chat(request.prompt.content)

            def after():
                exchange = Exchange(
                    prompt=request.prompt, reply=AssistantMessage(content=str(response))
                )
                self._model_cache[system, context + (exchange,)] = model

            response.after(after)
            return response

    return Service(ChatBotServiceProvider()) | model("chatbot")

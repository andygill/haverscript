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
    """A protocol for building chatbots.

    The assumption is that a chatbot looks after its own state.
    """

    def chat_to_bot(self, text: str) -> Reply:
        """take a string, update the object, and return a Reply."""
        ...


def connect_chatbot(
    chatbot: ChatBot | Callable[[str | None], ChatBot], name: str = "chatbot"
) -> Model:
    """promote a ChatBot into a Model.

    The argument can also be a function that takes an (optional) system prompt,
    then returns the ChatBot.

    connect_chatbot handles state by cloing the ChatBot automatically.
    """

    class ChatBotServiceProvider(ServiceProvider):
        def __init__(self):
            self._model_cache: dict[tuple[str, tuple[Exchange, ...]], ChatBot] = {}

        def list(self) -> list[str]:
            return [name]

        def ask(self, request: Request) -> Reply:
            assert isinstance(request, Request)
            assert isinstance(request.prompt, Prompt)
            assert isinstance(request.contexture, Contexture)

            system = request.contexture.system
            context = request.contexture.context

            try:
                _chatbot = self._model_cache[system, context]
            except KeyError:
                if context == ():
                    if callable(chatbot):
                        _chatbot = chatbot(system)
                    else:
                        assert system is None, "chatbot does not support system prompts"
                        _chatbot = deepcopy(chatbot)
                    self._model_cache[system, ()] = _chatbot
                else:
                    # We has a context we've never seen
                    # Which means we did not generate it
                    # Which means we reject it
                    # Should never happen with regular usage
                    assert False, "unknown system or context"

            _chatbot: ChatBot = deepcopy(_chatbot)

            response: Reply = _chatbot.chat_to_bot(request.prompt.content)

            def after():
                exchange = Exchange(
                    prompt=request.prompt, reply=AssistantMessage(content=str(response))
                )
                self._model_cache[system, context + (exchange,)] = _chatbot

            response.after(after)
            return response

    return Service(ChatBotServiceProvider()) | model(name)

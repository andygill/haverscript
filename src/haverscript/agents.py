from __future__ import annotations
from typing import Any, Type
import copy
from pydantic import BaseModel

from haverscript.haverscript import Model
from haverscript.markdown import Markdown
from haverscript.middleware import (
    EmptyMiddleware,
    format as format_middleware,
    stream as stream_middleware,
    echo as echo_middleware,
)
from haverscript.types import Middleware, EmptyMiddleware, Reply
from haverscript.markdown import markdown


class Agent(BaseModel):
    """An Agent is a python class that has access to an LLM.

    The subclass must provide a system prompt, called system,
    that describes the agent's role.

    The subclass may provide a prepare method that sets up
    the agent's state and other considerations.

    The model itself is immutable, so can be copied and used
    without side effects to the original.
    """

    model: Model

    def model_post_init(self, __context: dict | None = None) -> None:
        if self.system:
            self.model = self.model.system(markdown(self.system))
        self.prepare()

    def prepare(self):
        """Prepare the agent for a new conversation.

        This method is called before the agent is asked to chat.
        """
        pass

    def chat_llm(
        self,
        prompt: str | Markdown,
        format: Type | None = None,
        middleware: Middleware | None = None,
    ) -> str | Any:
        """ "chat with the llm and remember the conversation.

        If format is set, return the value of that type.
        """
        if middleware is None:
            middleware = EmptyMiddleware()
        if format is not None:
            middleware = format_middleware(format) | middleware

        response = self.model.chat(prompt, middleware=middleware)

        if format:
            return response.value

        return response.reply

    def ask_llm(
        self,
        prompt: str | Markdown,
        format: Type | None = None,
        middleware: Middleware | None = None,
        stream: bool = False,
    ) -> str | Any | Reply:
        """ask the llm something without recording the conversation.

        If stream is set, return a Reply object. Reply is a monad.

        If format is set, return the value of that type.

        Otherwise, return a string.
        """
        if middleware is None:
            middleware = EmptyMiddleware()
        if format is not None:
            middleware = format_middleware(format) | middleware
        if stream:
            middleware = stream_middleware() | middleware

        reply = self.model.ask(prompt, middleware=middleware)

        if stream:
            return reply

        if format:
            return reply.value

        return str(reply)

    def clone(self, kwargs: dict = {}) -> Agent:
        """clone the agent.

        The result should have its own identity and not be affected by the original agent.

        If the agent has additional state, it should have its own clone method,
        that calls this method with the additional state.
        """
        return type(self)(model=self.model, system=self.system, **kwargs)

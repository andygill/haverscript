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
from haverscript.tools import Tools
from haverscript.types import Middleware, EmptyMiddleware, Reply
from haverscript.markdown import markdown


class Agent(BaseModel):
    """An Agent is a python class that has access to an LLM.

    The subclass must provide a system prompt, called system,
    that describes the agent's role, or an implementation of
    prepare, which sets up the agent's state and other considerations.
    """

    model: Model

    def model_post_init(self, __context: dict | None = None) -> None:
        self.prepare()

    def prepare(self):
        """Prepare the agent for a new conversation.

        This method is called before the agent is asked to chat.
        This default adds self.system as a system command.
        self.system can be anything that can be str()ed.
        """
        self.model = self.model.system(str(self.system))

    def chat(
        self,
        prompt: str | Markdown,
        format: Type | None = None,
        middleware: Middleware | None = None,
        tools: Tools | None = None,
    ) -> str | Any:
        """ "chat with the llm and remember the conversation.

        If format is set, return the value of that type.
        """
        if middleware is None:
            middleware = EmptyMiddleware()
        if format is not None:
            middleware = format_middleware(format) | middleware

        response = self.model.chat(prompt, middleware=middleware, tools=tools)

        self.model = response

        if format:
            return response.value

        return response.reply

    def ask(
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

        reply: Reply = self.model.ask(prompt, middleware=middleware)

        if stream:
            return reply

        if format:
            return reply.value

        return str(reply)

    def remember(
        self,
        prompt: str | Markdown,
        reply: str | Reply,
    ) -> None:
        """remember a conversation exchange.

        This is used to add a conversation to the agent's memory.
        This is useful for advanced prompt techniques, such as
        chaining or tree of calls.
        """
        if isinstance(reply, str):
            reply = Reply([reply])

        self.model = self.model.process(prompt, reply)

    def clone(self, kwargs: dict = {}) -> Agent:
        """clone the agent.

        The result should have its own identity and not be affected by the original agent.

        If the agent has additional state, it should have its own clone method,
        that calls this method with the additional state.
        """
        return type(self)(model=self.model, system=self.system, **kwargs)

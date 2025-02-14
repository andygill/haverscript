from __future__ import annotations
from typing import Type
import copy
from pydantic import BaseModel

from haverscript.haverscript import Model
from haverscript.markdown import Markdown
from haverscript.middleware import EmptyMiddleware, format as format_middleware
from haverscript.types import Middleware, EmptyMiddleware
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
    persistence: bool = True

    def model_post_init(self, __context: dict | None = None) -> None:
        if self.system:
            self.model = self.model.system(markdown(self.system))
        self.prepare()

    def prepare(self):
        """Prepare the agent for a new conversation.

        This method is called before the agent is asked to chat.
        """
        pass

    def ask_llm(
        self,
        prompt: str | Markdown,
        format: Type | None = None,
        middleware: Middleware | None = None,
    ) -> str | Type:
        if middleware is None:
            middleware = EmptyMiddleware()
        if format is not None:
            middleware = format_middleware(format) | middleware

        response = self.model.chat(prompt, middleware=middleware)

        if self.persistence:
            # If this is a persistent agent, update the model with the exchange
            self.model = response

        if format:
            return response.value
        return response.reply

    def clone(self, kwargs: dict = {}) -> Agent:
        """clone the agent.

        The result should have its own identity and not be affected by the original agent.

        If the agent has additional state, it should have its own clone method,
        that calls this method with the additional state.
        """
        return type(self)(model=self.model, persistence=self.persistence, **kwargs)

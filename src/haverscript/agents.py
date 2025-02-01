from typing import Type
import copy
from haverscript.haverscript import Model
from haverscript.markdown import Markdown
from haverscript.middleware import EmptyMiddleware, format as format_middleware
from haverscript.types import Middleware, EmptyMiddleware


class Agent:
    """An Agent is a python class that has access to an LLM."""

    def __init__(self, model: Model, persistence: bool = True):
        self.model = model
        self.persistence = persistence

    def ask(
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

    def clone(self):
        """clone the agent.

        The result should have its own identity and not be affected by the original agent.
        """
        return copy.deepcopy(self)

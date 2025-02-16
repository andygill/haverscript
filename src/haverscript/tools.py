from __future__ import annotations
from typing import Callable
from pydantic import BaseModel, Field

from .utils import tool_schema


class Tools(BaseModel):
    pass

    def tool_schemas(self) -> tuple[dict, ...]:
        return ()

    def __add__(self, other: Tools) -> ToolPair:
        return ToolPair(lhs=self, rhs=other)

    def __call__(self, name, arguments):
        raise ValueError(f"Tool {name} not found")


class ToolPair(Tools):
    lhs: Tools
    rhs: Tools

    def tool_schemas(self) -> tuple[dict, ...]:
        return self.lhs.tool_schemas() + self.rhs.tool_schemas()

    def __call__(self, name, arguments):
        try:
            return self.lhs(name, arguments)
        except ValueError:
            return self.rhs(name, arguments)


class Tool(Tools):
    """Provide a tool to the LLM to optionally use.

    attributes:
        tool (Callable): The tool to provide to the LLM
        schema (dict): The schema for the tool
    """

    tool: Callable
    tool_schema: dict
    name: str
    debug: bool

    def tool_schemas(self):
        return (self.tool_schema,)

    def __call__(self, name, arguments):
        if self.name == name:
            return self.tool(**arguments)
        else:
            raise ValueError(f"Tool {name} not found")


def tool(function: Callable, schema: dict | None = None, debug: bool = False) -> Tool:
    """Provide a tool (a callback function) to the LLM to optionally use.

    args:
        function (Callable): The tool to provide to the LLM
        schema (dict | None): The schema for the tool. If none give, it will be inferred from the function signature and docstring
    """
    if schema is None:
        schema = tool_schema(function)

    name = schema["function"]["name"]

    return Tool(tool=function, tool_schema=schema, name=name, debug=debug)

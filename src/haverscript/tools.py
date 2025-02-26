from __future__ import annotations

import copy
import inspect
from collections.abc import Callable
from typing import Callable

import docstring_parser as dsp
from pydantic import BaseModel, Field, TypeAdapter


def tool_schema(tool: Callable):
    """
    Takes a callable tool, and returns a schema for the tool

    args:
        tool (Callable): The tool to get the schema for

    returns:
        dict: The schema for the tool
    """
    # check to see if tool is callable. If not raise an error
    if not callable(tool):
        raise TypeError("tool must be a callable")

    name = tool.__name__

    docstring = inspect.getdoc(tool)
    docstring_content = dsp.parse(docstring)
    description = docstring_content.short_description

    docstring_params = {}

    for param in docstring_content.params:
        docstring_params[param.arg_name] = param.description

    # get the signature of the tool
    signature = inspect.signature(tool)

    # get the parameters of the tool
    parameters = signature.parameters

    # create a dictionary to store the schema
    schema = {"type": "object", "required": [], "properties": {}}

    # loop through the parameters
    for parameter in parameters:
        parameter_object: inspect.Parameter = parameters[parameter]

        parameter_schema = copy.deepcopy(
            TypeAdapter(parameter_object.annotation).json_schema()
        )

        if parameter_object.default == inspect.Parameter.empty:
            schema["required"].append(parameter)

        if parameter in docstring_params:
            parameter_schema["description"] = docstring_params[parameter]

        schema["properties"][parameter] = parameter_schema

    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": schema},
    }


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

from collections.abc import Callable
import inspect
from pydantic import TypeAdapter
import docstring_parser as dsp
import copy


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

from haverscript import connect, echo

import time
import haverscript.together as together
from haverscript.tools import tool

# add_two_numbers and subtract_two_numbers taken from ollama tools example:
# https://github.com/ollama/ollama-python/blob/main/examples/tools.py


def add_two_numbers(a: int, b: int) -> int:
    """
    Add two numbers

    Args:
      a (int): The first number
      b (int): The second number

    Returns:
      int: The sum of the two numbers
    """
    return int(a) + int(b)


def subtract_two_numbers(a: int, b: int) -> int:
    """
    Subtract two numbers

    Args:
      a (int): The first number
      b (int): The second number

    Returns:
      int: The difference of the two numbers
    """
    return int(a) - int(b)


# Tools can still be manually defined and passed into the tool() middleware
subtract_two_numbers_schema = {
    "type": "function",
    "function": {
        "name": "subtract_two_numbers",
        "description": "Subtract two numbers",
        "parameters": {
            "type": "object",
            "required": ["a", "b"],
            "properties": {
                "a": {"type": "integer", "description": "The first number"},
                "b": {"type": "integer", "description": "The second number"},
            },
        },
    },
}

session = connect("llama3.1:8b") | echo()
session2 = together.connect("meta-llama/Llama-3.3-70B-Instruct-Turbo") | echo(
    stream=False
)

session = session.system(
    """You are a helpful assistant that can access external functions. """
    """The responses from these function calls will be appended to this dialogue. """
    """Please provide responses based on the information from these function calls."""
)


tools = tool(add_two_numbers) + tool(
    subtract_two_numbers, schema=subtract_two_numbers_schema
)

session = session.chat("4712 plus 4734 is?", tools=tools)
session = session.chat("take this result, and subtract 1923", tools=tools)

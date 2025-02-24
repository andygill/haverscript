from enum import Enum
import inspect
import json
from typing import Type, Callable, Any, get_type_hints, get_args, get_origin
import string
import re
from pydantic import BaseModel


class Markdown:
    def __init__(
        self, content: list[str] | None = None, args: list[set[str]] | None = None
    ):
        if content is None:
            self.blocks = []
        else:
            self.blocks = content

        if args is None:
            self.args = [set() for _ in self.blocks]
        else:
            self.args = args

        assert len(self.blocks) == len(self.args), "internal inconsistency in Markdown"

    def __str__(self):
        # all blocks are separated by a blank line
        # this does not do any formatting of {variables}.
        return "\n\n".join(self.blocks)

    def format(self, **kwargs: dict):
        return "\n\n".join(
            [
                (
                    block.format(
                        **{key: kwargs[key] for key in spec_keys if key in kwargs},
                    )
                    if spec_keys
                    else block
                )
                for block, spec_keys in zip(self.blocks, self.args)
            ]
        )

    def __add__(self, other):
        if isinstance(other, Markdown):
            return Markdown(self.blocks + other.blocks, self.args + other.args)
        txt = str(other)
        txt = strip_text(txt)
        return Markdown(self.blocks + [txt], self.args + [set()])


def markdown(content: str | Markdown | None = None) -> Markdown:
    """Convert a string or markdown block into markdown block."""
    if content is None:
        return Markdown()
    if isinstance(content, str):
        return text(content)
    return content


def strip_text(txt: str) -> str:
    txt = txt.strip()  # remove blank lines before and after
    txt = "\n".join([line.strip() for line in txt.splitlines()])
    return txt


def header(txt, level: int = 1) -> Markdown:
    """Return a markdown header."""
    return Markdown([f"{'#' * level} {txt}"])


def xml_element(tag: str, contents: str | Markdown | None = None):
    singleton = Markdown([f"<{tag}/>"])

    if contents is None:
        return singleton

    inner: Markdown = markdown(contents)

    if len(inner.blocks) == 0:
        return singleton

    blocks = inner.blocks
    blocks = [f"<{tag}>\n{blocks[0]}"] + blocks[1:]
    blocks = blocks[:-1] + [f"{blocks[-1]}\n</{tag}>"]

    return Markdown(content=blocks, args=inner.args)


def bullets(items, ordered: bool = False) -> Markdown:
    """Return a markdown bullet list."""
    if ordered:
        markdown_items = [f"{i+1}. {item}" for i, item in enumerate(items)]
    else:
        markdown_items = [f"- {item}" for item in items]
    return Markdown(["\n".join(markdown_items)])


def code(code: str, language: str = "") -> Markdown:
    """Return a markdown code block."""
    return Markdown([f"```{language}\n{code}\n```"])


def text(txt: str) -> Markdown:
    """Return a markdown text block.

    Note that when using + or +=, text is automatically converted to a markdown block.
    """
    return Markdown() + txt


def quoted(txt) -> Markdown:
    """Return a quotes text block inside triple quotes."""
    return Markdown([f'"""\n{txt}\n"""'])


def rule(count: int = 3) -> Markdown:
    """Return a markdown horizontal rule."""
    return Markdown(["-" * count])


def table(headers: dict, rows: list[dict]) -> Markdown:
    """Return a markdown table.

    headers is a dictionary of column names to display.
    rows is a list of dictionaries, each containing the data for a row.

    We right justify integers and left justify anything else.
    """
    col_widths = {
        key: max(*(len(str(row[key])) for row in [headers] + rows)) for key in headers
    }

    separator = "|-" + "-|-".join("-" * col_widths[key] for key in headers) + "-|"

    header_row = (
        "| " + " | ".join(f"{headers[key]:{col_widths[key]}}" for key in headers) + " |"
    )
    data_rows = [
        "| "
        + " | ".join(
            (
                f"{row[key]:>{col_widths[key]}}"
                if isinstance(row[key], int)
                else f"{row[key]:<{col_widths[key]}}"
            )
            for key in headers
        )
        + " |"
        for row in rows
    ]
    return "\n".join([header_row, separator] + data_rows)


def reply_in_json(
    model: Type[BaseModel], prefix: str = "Reply in JSON, using the following keys:"
) -> Markdown:
    """Instructions to reply in JSON using a list of bullets schema."""
    prompt = Markdown()
    prompt += text(prefix)
    items = []
    type_hints = get_type_hints(model)
    for key, value in model.model_fields.items():
        annotation = type_hints.get(key, value.annotation)
        if annotation is None:
            type_name = ""
        elif get_origin(annotation) is list and get_args(annotation):
            type_name = f" (list of {get_args(annotation)[0].__name__})"
        elif inspect.isclass(annotation) and issubclass(annotation, Enum):
            enum_values = " or ".join(json.dumps(item.value) for item in annotation)
            type_name = f" ({enum_values})"
        elif isinstance(annotation, type):
            type_name = f" ({annotation.__name__})"
        else:
            type_name = (
                f" ({repr(annotation).replace('typing.', '').replace('|', 'or')})"
            )
        items.append(f'"{str(key)}"{type_name}: {value.description}')
    prompt += bullets(items)
    return prompt


def template(fstring: str) -> Markdown:
    """Delay interpretation of a f-string until later.

    variables are allowed inside {braces}, and will be filled in
    by calls to the format method.
    """
    formatter = string.Formatter()
    variables = set()
    fstring = strip_text(fstring)

    for _, field_name, _, _ in formatter.parse(fstring):
        if field_name:
            variables.add(field_name)
            assert bool(
                re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", field_name)
            ), f"invalid variable expression {field_name}, should be a variable name."

    return Markdown([fstring], [variables])

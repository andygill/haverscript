from typing import Type

from pydantic import BaseModel


class Markdown:
    def __init__(self, content: str | list[str] | None = None):
        if content is None:
            self.blocks = []
        elif isinstance(content, str):
            self.blocks = [content]
        else:
            self.blocks = content

    def __str__(self):
        # all blocks are separated by a blank line
        return "\n\n".join(self.blocks)

    def __add__(self, other):
        if isinstance(other, Markdown):
            return Markdown(self.blocks + other.blocks)
        txt = str(other)
        txt = txt.strip()  # remove blank lines before and after
        txt = "\n".join([line.strip() for line in txt.splitlines()])
        return Markdown(self.blocks + [txt])


def header(txt, level: int = 1) -> Markdown:
    """Return a markdown header."""
    return Markdown(f"{'#' * level} {txt}")


def bullets(items, ordered: bool = False) -> Markdown:
    """Return a markdown bullet list."""
    if ordered:
        markdown_items = [f"{i+1}. {item}" for i, item in enumerate(items)]
    else:
        markdown_items = [f"- {item}" for item in items]
    return Markdown("\n".join(markdown_items))


def code(code: str, language: str = "") -> Markdown:
    """Return a markdown code block."""
    return Markdown(f"```{language}\n{code}\n```")


def text(txt: str) -> Markdown:
    """Return a markdown text block.

    Note that when using + or +=, text is automatically converted to a markdown block.
    """
    return Markdown() + txt


def quoted(txt) -> Markdown:
    """Return a quotes text block inside triple quotes."""
    return Markdown(f'"""\n{txt}\n"""')


def rule(count: int = 3) -> Markdown:
    """Return a markdown horizontal rule."""
    return Markdown("-" * count)


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
    prompt += bullets(
        [
            f'"{str(key)}" ({value.annotation}): {value.description}'
            for key, value in model.model_fields.items()
        ]
    )
    return prompt

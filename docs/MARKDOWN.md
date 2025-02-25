# Markdown DSL Documentation

The Markdown DSL is a lightweight domain-specific language for programmatically
creating Markdown content for prompts. It provides an readable API for
constructing Markdown documents by combining text blocks, headers, lists, code
blocks, tables, and more. The typical idiom for using the library is:

```python
prompt = Markdown()

prompt += ...
prompt += ...

# use the constructed prompt
```

Below is a comprehensive guide to the API and usage examples.

## Overview

The core of the library is the `Markdown` class, which internally maintains a
list of markdown blocks. Blocks are joined by blank lines when rendering the
document. The library also provides helper functions to create various markdown
elements such as headers, bullet lists, code blocks, tables, XML elements, and
more. Additionally, it supports delayed evaluation of f-strings using a template
mechanism, which allows you to format variable placeholders later.

---

## Getting Started

To get started, create a `Markdown` object and use the `+=` operator to append
content. When you append a string, it is automatically converted into a markdown
block. Finally, render your document by converting the `Markdown` object to a
string or by using the `format` method for variable substitution.

**Basic Usage Example:**

```python
prompt = Markdown()
prompt += header("Welcome to My Document")
prompt += text("This document is generated using a Markdown DSL.")
prompt += bullets(["First item", "Second item", "Third item"])
print(prompt)
```

If you have variables within your text, use `template` and the `format` method to substitute them:

```python
prompt += template("Hello, {name}!")
print(prompt.format(name="Alice"))
```

---

## API Reference

### Class: `Markdown`

#### Initialization
```python
Markdown()
```
  
Creates a new Markdown object. 

#### Methods

- **`__str__`**  
  Renders the complete document as a string. Each block is separated by a blank line.

- **`format(**kwargs)`**  
  Returns a formatted version of the markdown document by replacing variables (e.g., `{variable}`) in each block with provided keyword arguments.

- **`__add__(other)`**  
  Supports appending content to the Markdown document. When a string is added, it is automatically converted to a Markdown block.

  This provides the support for `+` and `=+` for markdown blocks.

---

### Helper Functions

#### `markdown(content: str | Markdown | None = None) -> Markdown`
Converts a string or an existing Markdown block into a Markdown object.
- If `content` is `None`, returns an empty Markdown object.
- If `content` is a string, it is converted using the `text` function.

---

#### `strip_text(txt: str) -> str`
Utility function that:
- Strips leading and trailing whitespace.
- Trims whitespace from each line.
  
Useful for cleaning input text before adding it to a Markdown block.

---

#### `header(txt, level: int = 1) -> Markdown`
Creates a Markdown header.
- **txt**: The header text.
- **level**: The header level (default is 1). Uses the `#` symbol repeated `level` times.

Example:
```python
header("Introduction", level=2)  # Produces "## Introduction"
```

---

#### `xml_element(tag: str, contents: str | Markdown | None = None) -> Markdown`
Wraps the given content in an XML element.
- **tag**: The XML tag name.
- **contents**: Content to be wrapped. If omitted or empty, returns a self-closing tag (`<tag/>`).

If content is provided, the element is formatted as:
```xml
<tag>
... content ...
</tag>
```

---

#### `bullets(items, ordered: bool = False) -> Markdown`
Generates a bullet list.
- **items**: An iterable of items to include in the list.
- **ordered**: If `True`, creates an ordered (numbered) list; otherwise, an unordered list using hyphens.

Example:
```python
bullets(["Item 1", "Item 2"], ordered=True)
```

---

#### `code(code: str, language: str = "") -> Markdown`
Creates a Markdown code block.
- **code**: The code snippet.
- **language**: The programming language for syntax highlighting (optional).

Example:
```python
code("print('Hello, world!')", language="python")
```

---

#### `text(txt: str) -> Markdown`
Creates a Markdown text block from a given string. When using the `+=` operator, strings are automatically converted using this function.

---

#### `quoted(txt) -> Markdown`
Wraps the provided text inside triple quotes (`""" ... """`), designating it as a quoted block.

---

#### `rule(count: int = 3) -> Markdown`
Generates a horizontal rule.
- **count**: The number of dashes to use (default is 3).

Example:
```python
rule()  # Produces '---'
```

---

#### `table(headers: dict, rows: list[dict]) -> Markdown`
Creates a Markdown table.
- **headers**: A dictionary mapping column keys to header names.
- **rows**: A list of dictionaries, each representing a row in the table.

The function automatically right-justifies integers and left-justifies other types.

Example:
```python
headers = {"name": "Name", "age": "Age"}
rows = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
table_md = table(headers, rows)
```

---

#### `reply_in_json(model: Type[BaseModel], prefix: str = "Reply in JSON, using the following keys:") -> Markdown`
Generates instructions for replying in JSON format based on a Pydantic model.
- **model**: A Pydantic `BaseModel` subclass defining the schema.
- **prefix**: A prefix instruction string.

It introspects the model’s type hints and field descriptions to produce a bullet list of keys along with type information. This is of more utility to an LLM that the raw json schema.

Example:
```python
from pydantic import BaseModel, Field
from enum import Enum

class Status(Enum):
    SUCCESS = "success"
    FAILURE = "failure"

class ReplyModel(BaseModel):
    status: Status = Field(description="The status of the operation")
    message: str = Field(description="Detailed message regarding the result")

prompt = reply_in_json(ReplyModel)
```

---

#### `template(fstring: str) -> Markdown`
Delays the interpretation of an f-string until later.
- **fstring**: A string containing variable placeholders (e.g., `"Hello, {name}!"`).

The function extracts valid variable names and stores them so that they can later be substituted using the `format` method.

Example:
```python
prompt = template("Hello, {username}! Welcome to our service.")
formatted_prompt = prompt.format(username="Alice")
```

---

## Usage Examples

### 1. Building a Simple Document

```python
prompt = Markdown()
prompt += header("Document Title", level=4)
prompt += text("Welcome to the generated markdown document.")
prompt += bullets(["Item 1", "Item 2", "Item 3"])
prompt += code("print('Hello, world!')", language="python")
print(prompt)
```

> #### Document Title
> 
> Welcome to the generated markdown document.
>
> - Item 1
> - Item 2
> - Item 3
>
> ```python
> print('Hello, world!')
> ```

### 2. Delayed Formatting with Template

```python
prompt = Markdown()
prompt += header("User Greeting", level=4)
prompt += template("Hello, {username}! Welcome to our service.")
formatted_prompt = prompt.format(username="Bob")
print(formatted_prompt)
```

> #### User Greeting
>
> Hello, Bob! Welcome to our service.

### 3. Creating a Table

```python
headers = {"name": "Name", "age": "Age"}
rows = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
table_md = table(headers, rows)
print(table_md)
```

> | Name  | Age |
> |-------|-----|
> | Alice |  30 |
> | Bob   |  25 |

### 4. Generating a JSON Reply Prompt

```python
from pydantic import BaseModel, Field
from enum import Enum

class Status(Enum):
    SUCCESS = "success"
    FAILURE = "failure"

class ReplyModel(BaseModel):
    status: Status = Field(description="The status of the operation")
    message: str = Field(description="Detailed message regarding the result")

prompt = reply_in_json(ReplyModel)
print(prompt)
```

> Reply in JSON, using the following keys:
> 
> - "status" ("success" or "failure"): The status of the operation
> - "message" (str): Detailed message regarding the result

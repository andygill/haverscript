# Markdown Prompts

Haverscript provides basic support for markdown-style prompts. Both `.chat()`
and `.system()` automatically convert `Markdown` to `str`, as needed.

Here is an example of use. The idiom is you create an empty `Markdown()`,
then append to this to create the markdown document.

```python
from haverscript import Markdown, header, bullets

prompt = Markdown()
prompt += header("Background")
prompt += "Some background for the LLM"
prompt += header("Instructions")
prompt += "Please write some text"
prompt += bullets(["Remember this", "Remember this as as well"])
```

This will give the following output (extracting using `print(prompt)`):

```
# Background

Some background for the LLM

# Instructions

Please write some text

- Remember this
- Remember this as as well
```

The markdown support also includes `quoted` strings, `code` blocks, and `table` support.

```
from haverscript import Markdown, table

prompt = Markdown()
prompt += table(
    {"name": "Name", "age": "Age"},
    [
        {"name": "John", "age": 20},
        {"name": "Paul", "age": 22},
        {"name": "George", "age": 23},
        {"name": "Ringo", "age": 24},
    ],
)
```

This will give the contents for `prompt`:

```
| Name   | Age |
|--------|-----|
| John   |  20 |
| Paul   |  22 |
| George |  23 |
| Ringo  |  24 |
```

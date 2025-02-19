# Markdown Prompts

Haverscript provides basic support for markdown-style prompts. `.chat()`, 
`.ask()` and `.system()` automatically convert `Markdown` to `str`, as needed.

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

```python
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

When using structured output, a simple schema can be auto-generated
for the LLM to use.

```python
class EditorFeedback(BaseModel):
    comments: list[str] = Field(
        default_factory=list,
        description="Specific concrete recommendation of improvement",
    )
    score: int = Field(..., description="Quality score from 1 to 10")


prompt = Markdown()
prompt += "please give suggestions for improvement"
prompt += reply_in_json(EditorFeedback)
```

This will give the contents for `prompt`:

```
please give suggestions for improvement

Reply in JSON, using the following keys:
- "comments" (list of str): Specific concrete recommendation of improvement
- "score" (int): Quality score from 1 to 10
```

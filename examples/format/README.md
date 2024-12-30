Often, you want the LLM to return its response in a structured manner. There is
support in most LLMs for enabling JSON output, and even enabling JSON output
that complies with a given schema.

We use the `format()` middleware, to both request
JSON, and supply a schema (via the pydantic BaseModel).

```python
from pydantic import BaseModel

from haverscript import connect, format


class Translate(BaseModel):
    english: str
    french: str


model = connect("mistral")

prompt = (
    f"You are a translator, translating from English to French. "
    f'Give your answer as a JSON record with two fields, "english", and "french". '
    f'The "english" field should contain the original English text, and only the original text.'
    f'The "french" field should contain the translated French text, and only the translated text.'
    f"\n\nEnglish Text: Three Blind Mice"
)

print(model.chat(prompt, middleware=format(Translate)).value)
```

When run, this gives the following reply:

```python
english='Three Blind Mice' french='Trois Souris Aveugles'
```
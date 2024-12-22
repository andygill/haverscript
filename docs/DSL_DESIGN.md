# Haverscript Domain Specific Language

First, let's consider several different Python APIs for accessing LLMs.

### Using Ollama's Python API

From <https://github.com/ollama/ollama-python>


```python
import ollama

stream = ollama.chat(
    model='llama3.1',
    messages=[
        {
            'role': 'user', 
            'content': 'Why is the sky blue?'
        }
    ],
    stream=True,
)

for chunk in stream:
  print(chunk['message']['content'], end='', flush=True)
```

### Using Mistral's Python API

From <https://docs.mistral.ai/getting-started/clients/>

```python
import os
from mistralai import Mistral

api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-large-latest"

client = Mistral(api_key=api_key)

chat_response = client.chat.complete(
    model = model,
    messages = [
        {
            "role": "user",
            "content": "What is the best French cheese?",
        },
    ]
)

print(chat_response.choices[0].message.content)
```

### Using LiteLLM's Python API

From <https://github.com/BerriAI/litellm>

```python
from litellm import acompletion
import asyncio

async def test_get_response():
    user_message = "Hello, how are you?"
    messages = [{"content": user_message, "role": "user"}]
    response = await acompletion(model="gpt-3.5-turbo", messages=messages)
    return response

response = asyncio.run(test_get_response())
print(response)
```

### LangChain (Using Ollama)

From <https://python.langchain.com/v0.2/docs/integrations/llms/ollama/>,
with output parser from <https://python.langchain.com/v0.1/docs/get_started/quickstart/>.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_core.output_parsers import StrOutputParser

template = """Question: {question}

Answer: Let's think step by step."""

prompt = ChatPromptTemplate.from_template(template)

model = OllamaLLM(model="llama3.1")

output_parser = StrOutputParser()

chain = prompt | model | output_parser

chain.invoke({"question": "What is LangChain?"})
```

All these examples are more verbose than necessary for such simple tasks.

## A Domain Specific Language for talking to LLMs

A Domain Specific Language (DSL) is an interface to a capability. A good LLM DSL
should have composable parts that work together to make accessing the
capabilities of LLMs robust, straightforward, and predictable.

So, when we want to connect to a specific LLM, ask a question, and print a
result, we do the following:

```python
from haverscript import connect

print(connect("mistral").chat("What is a haver?"))
```

---

What about having a chat session? Turn on echo, and chat away. We turn on echo
by using a pipe after the `connect`, and piping the response into `echo`.

```python
from haverscript import connect

session = connect("mistral") | echo()
session = session.chat("In one sentence, why is the sky blue?")
session = session.chat("Rewrite the above sentence in the style of Yoda")
session = session.chat("How many questions did I ask?")
...
```

---

What about using the result in a later prompt? Use a Python f-string to
describe the second prompt.

```python
from haverscript import connect

model = connect("mistral") | echo()

best = model.chat("Name the best basketball player. Only name one player and do not give commentary.")

model.chat(f"Someone told me that {best} is the best basketball player. Do you agree, and why?")
```

These examples demonstrate how Haverscript's components are designed to be
composable, making it easy to build upon previous interactions.

## Principles of Haverscript as a DSL

Haverscript is built around four core principles:

**LLM Interactions as String-to-String Operations**

Interactions with LLMs in Haverscript are fundamentally string-based. Python’s
robust string formatting tools, such as  `.format` and f-strings, are used
directly. Prompts can be crafted using f-strings with explicit `{...}` notation
for injection. The `chat` method accepts prompt strings and returns a result
that can be seamlessly used in other f-strings, or accessed through the `.reply`
attribute. This makes the "string plumbing" for prompt-based applications an
integral part of the Haverscript domain-specific design.

**Immutable Core Structures**

The user-facing classes in Haverscript are immutable, similar to Python strings or
tuples. Managing state is as simple as assigning names to things. For example,
running the same prompt multiple times on the same context is straightforward
because there is no hidden state that might be updated.

**Chat as a Chain of Responses**

Using immutable structures, `.chat` is a chain of responses, taking care of the
context of calls behind the scenes. Calling `.chat` both calls the LLM, and builds
the complete context needed to make this call. 

**Middleware for Effects**

Haverscript provides extensions to the basic chat using middleware and a pipe syntax.
Things between the call to chat and the call to the actual LLM service can be augmented.
These include Caching (using SQLite), retry and validation, echo capability, and
The user can build the pipeline between the call to `.chat`, and the call of the LLM,
that works for them.

For example, if we want to both echo and use a cache, then there are two choices:
echo every response, or echo only the non-cached resposes. The user picks what works
for them using composable parts.


from haverscript import echo, options
from haverscript.together import connect

file = "README.md"

prompt = """
Make this sentence shorter and crisper.
It is the first sentence of an introduction.
---
Haverscript is a lightweight Python library designed to manage Large Language Model (LLM) interactions.
""".strip()


session = (
    connect("meta-llama/Llama-3.3-70B-Instruct-Turbo")
    | echo(prompt=False, spinner=False, width=100)
    | options(temperature=1.6)
)
for i in range(10):
    temperature = 0.8 + i / 10
    print(f"\ntemperature={temperature}\n----------------")
    for _ in range(3):
        session.chat(prompt, middleware=options(temperature=temperature))

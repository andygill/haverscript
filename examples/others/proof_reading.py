from haverscript import echo, options, dedent
from haverscript.together import connect


file = "README.md"

prompt = f"""
Here is some markdown that is the README for a LLM-based agent language called HaverScript.
Please make last minute suggestions (typos, grammar, etc.), using a numbered list to do so,
in the same order as the items appear in the README.
After each item, list the specific changes you would like to see (as a diff if possible).

The code and README.md goes out today.
---
{open(file).read()}
"""

session = (
    connect("meta-llama/Llama-3.3-70B-Instruct-Turbo")
    | echo(prompt=False, spinner=False)
    | options(temperature=0.6)
    | dedent()
)
session.chat(prompt)

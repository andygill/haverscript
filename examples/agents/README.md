Haverscript has support for basic agents. An `Agent` is a class that has access
to an LLM, via the `Model` class. However, an `Agent` also has state, and acts
as an object (vs a structure).

Think of an Agent as a python object that can take care of something. So in this
example, we have two agents, an `Author` and an `Editor`.

## Author agent

Now, the only requirement of an agent is to define a system prompt as
an attribute (but see Notes below for additional options).

```python
class Author(Agent):
    system: str = """
    You are a world-class author who is writing a travel book.

    You are part of a team.

    Instructions are given with the heading "# Instructions". Respond in the requested format.

    Commentary, feedback or requests from others are given with the heading "# Feedback from ..."
    """
```

Further, an agent should have a capability, in this case writing.

```python
    def write(
        self, instructions: str | Markdown, feedback: EditorFeedback | None = None
    ) -> str:
        prompt = Markdown()

        if feedback is not None:
            prompt += header("Feedback from Editor")
            prompt += bullets(feedback.comments)

        prompt += header("Instructions") + instructions

        return self.chat(prompt)
```


Here,we take the instructions, and feedback from our editor, and call the LLM.

## Editor agent.

Again, we define the agent.

```python
class Editor(Agent):
    system: str = """
    You are a editor for a company that writes travel books.
    Make factual and actionable suggestions for improvement.
    """
```

This time, we have the capability for proofing text.

```python
    def proof(self, instructions: str, article: str) -> EditorFeedback:
        prompt = Markdown()
        prompt += header("Previous Text")
        prompt += "Original instructions given to Author:"
        prompt += quoted(instructions)

        ...

        prompt += reply_in_json(EditorFeedback)

        return self.ask(prompt, format=EditorFeedback)
```

We reqire the output use the `EditorFeedback` class.

```python
class EditorFeedback(BaseModel):
    comments: list[str] = Field(
        default_factory=list,
        description="Specific concrete recommendation of improvement",
    )
    score: int = Field(..., description="Quality score from 1 to 10")
```

Now, we call `ask_llm` this time - ask is for calls without history,
chat is for sessions with history - and we want the editor to consider
what is written in isolation.

Thats it! We've written two agents.

We connect the two models using a supervisor class.

```python
class Supervision(BaseModel):
    author: Author
    editor: Editor
    """Generic class with author and editor interacting."""
```

If we instatiate this, we can now use the supervisor to writing
and improve text.


```python
prompt = template(
    """Write {words} words about traveling to Scotland. Only write prose. No titles or lists."""
)
print(supervised.improve(
    "Scotland.",
    prompt,
    ({"words": words} for words in [200, 300, 400]),
))
```

This writes articles of 200, then 300, then 400 words, each time geting feedback,
and improving the next version. This is a dynamic form of multishot prompting,
but using agents to do the plumbing.

# Notes

- There is a way, by providing an `Agent.prepare()` method, to have
dynamic system prompts. This is an advanced topic.

- The ask_llm can return a `Reply`, and allow agents to do work in 
a way to get dynamic output as LLMs are called.

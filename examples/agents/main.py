from haverscript import (
    connect,
    echo,
    options,
    Agent,
    Markdown,
    bullets,
    text,
    header,
    text,
    quoted,
    reply_in_json,
)

from typing import Any, Iterable
from pydantic import BaseModel, Field
from typing import Callable


model_name = "mistral-nemo:latest"

model = connect(model_name) | options(num_ctx=16 * 1024)


class EditorFeedback(BaseModel):
    comments: list[str] = Field(
        default_factory=list,
        description="Specific concrete recommendation of improvement",
    )

    def __str__(self) -> str:
        response = Markdown()
        response += bullets(self.comments)
        return str(response)


class Author(Agent):
    system = """
    You are a world-class author who is writing a travel book.

    You are part of a team.

    Instructions are given with the heading "# Instructions". Respond in the requested format.

    Commentary, feedback or requests from others are given with the heading "# Feedback from ..."
    """

    def __init__(self):
        # We use markdown's text to clean up the system prompt, and turn on echo
        super().__init__(model.system(text(self.system)) | echo())

    def __call__(self, instructions: str, feedback: EditorFeedback | None = None):
        prompt = Markdown()

        if feedback is not None:
            prompt += header("Feedback from Editor")
            prompt += bullets(feedback.comments)

        prompt += header("Instructions") + instructions
        return self.ask(str(prompt))


class Editor(Agent):
    system = """
    You are a editor for a company that writes travel books. Make factual and actionable
    suggestions for improvement.
    """

    def __init__(self):
        super().__init__(model.system(text(self.system)) | echo(), persistence=False)

    def __call__(self, instructions: str, article: str) -> EditorFeedback:
        prompt = Markdown()
        prompt += header("Previous Text")
        prompt += "Instructions for Author:"
        prompt += quoted(instructions)
        prompt += "Text from Author: "
        prompt += quoted(article)

        prompt += header("Validation")

        prompt += f"""
        Read the above Text, and consider the following criteria:

            - Is the text engaging and informative?
            - Does the text have a clear structure?
            - Are the sentences well-constructed?
            - Are there any factual inaccuracies?
            - Are there any spelling or grammar mistakes?
            - Are there any areas that could be improved?

        Given these criteria, and the original instructions,
        assess the text and provide specific feedback on how it could be improved."""

        prompt += reply_in_json(EditorFeedback)

        return self.ask(str(prompt), format=EditorFeedback)


author = Author()
editor = Editor()


class Supervision:
    """Generic class with author and editor interacting."""

    def __init__(self, author, editor):
        self.author = author
        self.editor = editor

    def __call__(
        self,
        instructions: str | Markdown | Callable[[Any], str | Markdown],
        rounds: Iterable[int],
    ):
        feedback = None
        first_round = True
        editor = self.editor()
        author = self.author()
        for round in rounds:
            if first_round:
                feedback = None
            else:
                feedback = editor(prompt, article)

            if callable(instructions):
                prompt = instructions(round)
            else:
                prompt = instructions

            article = author(prompt, feedback)
            first_round = False


supervision = Supervision(Author, Editor)

supervision(
    lambda words: f"Consider the feedback above, previous attempts to write about Scotland (if any). "
    f"Now write {words} words about traveling to Scotland. Only write prose. No titles or lists.",
    [200, 300, 400, 500],
)

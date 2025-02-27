from haverscript import (
    connect,
    options,
    Agent,
    Markdown,
    bullets,
    header,
    quoted,
    reply_in_json,
    template,
    stats,
)

from typing import Iterable, Iterator
from pydantic import BaseModel, Field


model_name = "mistral-nemo:latest"

model = connect(model_name) | options(num_ctx=16 * 1024, temperature=0.3) | stats()


class EditorFeedback(BaseModel):
    comments: list[str] = Field(
        default_factory=list,
        description="Specific concrete recommendation of improvement",
    )
    score: int = Field(..., description="Quality score from 1 to 10")

    def __str__(self) -> str:
        response = Markdown()
        response += bullets(self.comments)
        response += f"Quality score out of 10: {self.score}"
        return str(response)


class Author(Agent):
    system: str = """
    You are a world-class author who is writing a travel book.

    You are part of a team.

    Instructions are given with the heading "# Instructions". Respond in the requested format.

    Commentary, feedback or requests from others are given with the heading "# Feedback from ..."
    """

    def write(
        self, instructions: str | Markdown, feedback: EditorFeedback | None = None
    ) -> str:
        prompt = Markdown()

        if feedback is not None:
            prompt += header("Feedback from Editor")
            prompt += bullets(feedback.comments)

        prompt += header("Instructions") + instructions

        return self.chat(prompt)


class Editor(Agent):
    system: str = """
    You are a editor for a company that writes travel books.
    Make factual and actionable suggestions for improvement.
    """

    def proof(self, instructions: str, article: str) -> EditorFeedback:
        prompt = Markdown()
        prompt += header("Previous Text")
        prompt += "Original instructions given to Author:"
        prompt += quoted(instructions)
        prompt += "Text from Author following original instructions: "
        prompt += quoted(article)

        prompt += header("Instructions")

        prompt += "Read the above Text, and consider the following criteria:"
        prompt += bullets(
            [
                "Does the text follow the original instructions?",
                "Is the text engaging and informative?",
                "Does the text have a clear structure?",
                "Are the sentences well-constructed?",
                "Are there any factual inaccuracies?",
                "Are there any spelling or grammar mistakes?",
                "Are there any areas that could be improved?",
            ]
        )

        prompt += """
            Given these criteria, and the original instructions,
            assess the text and provide specific feedback on how it could be improved.
            Also give a numerical score from 1 to 10, where 1 is the worst and 10 is the best,
            regarding quality and suitability for a travel book.
        """

        prompt += reply_in_json(EditorFeedback)

        return self.ask(prompt, format=EditorFeedback)


class Supervision(BaseModel):
    author: Author
    editor: Editor
    """Generic class with author and editor interacting."""

    def improve(
        self,
        topic: str | Markdown,
        instructions: str | Markdown,
        bindings: Iterable[dict[str, str]],
    ) -> Iterator[str]:
        feedback = None
        first_round = True
        editor = self.editor.clone()
        author = self.author.clone()
        formated_instructions = None
        for binding in bindings:
            prompt = Markdown()

            if formated_instructions:
                feedback = editor.proof(formated_instructions, article)
                prompt += f"Consider the feedback from the Editor, and your previous attempt to write about {topic}."
            else:
                feedback = None

            formated_instructions = instructions.format(**binding)
            prompt += formated_instructions

            article = author.write(prompt, feedback)
            first_round = False

        return article


supervised = Supervision(author=Author(model=model), editor=Editor(model=model))

prompt = template(
    """Write {words} words about traveling to Scotland. Only write prose. No titles or lists."""
)

article = Author(model=model).write(prompt.format(words=400))
print("Zero-shot Article:")
print(article)

article = supervised.improve(
    "Scotland.",
    prompt,
    ({"words": words} for words in [200, 300, 400]),
)
print("Article using supervision:")
print(article)

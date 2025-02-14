from pydantic import BaseModel, Field
from typing import Iterator

from haverscript import *
from haverscript.types import *
from haverscript.middleware import *


class Translate(BaseModel):
    english: str = Field(
        ..., description="the original English text, and only the original text."
    )
    french: str = Field(
        ..., description="the translated French text, and only the translated text."
    )


class MetaAgent(Agent, ServiceProvider):
    system: str = "You are a translator, translating from English to French. "
    previous: list[Translate] = Field(default_factory=list)

    def list(self) -> list[str]:
        return ["french"]

    def ask(self, request: Request) -> Reply:
        assert isinstance(request, Request)
        assert isinstance(request.prompt, Prompt)
        assert isinstance(request.contexture, Contexture)

        return Reply(self._stream(request))

    def _stream(self, request) -> Iterator:
        max_traslations = 3
        if len(self.previous) >= 3:
            yield f"Sorry. French lesson over.\nYour {max_traslations} translations:\n"
            for resp in self.previous:
                assert isinstance(resp, Translate)
                yield f"* {resp.english} => {resp.french}\n"
        else:
            prompt = Markdown()
            prompt += f"You are a translator, translating from English to French. "
            prompt += f"English Text: {request.prompt.content}"
            prompt += reply_in_json(Translate)

            translated: Translate = self.ask_llm(prompt, format=Translate)

            self.previous.append(translated)
            remaining = max_traslations - len(self.previous)

            yield f"{translated.english} in French is {translated.french}\n"
            yield "\n"
            yield f"{remaining} translation(s) left."


# In a real example, validate and retry would be added to provide robustness.
session = Service(MetaAgent(model=connect("mistral"))) | model("french") | echo()

print("--[ User-facing conversation ]------")
session = session.chat("Three blind mice")
session = session.chat("Such is life")
session = session.chat("All roads lead to Rome")
session = session.chat("The quick brown fox")

session = Service(MetaAgent(model=connect("mistral") | echo())) | model("french")

print("--[ LLM-facing conversation ]------")
session = session.chat("Three blind mice")
session = session.chat("Such is life")
session = session.chat("All roads lead to Rome")
session = session.chat("The quick brown fox")

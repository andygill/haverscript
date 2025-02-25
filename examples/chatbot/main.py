from pydantic import BaseModel, Field
from typing import Iterator, Protocol, Callable

from haverscript import *
from haverscript.types import Reply, Prompt, Contexture
from copy import deepcopy


class Translate(BaseModel):
    english: str = Field(
        ..., description="the original English text, and only the original text."
    )
    french: str = Field(
        ..., description="the translated French text, and only the translated text."
    )


class FrenchAgent(Agent):
    system: str = "You are a translator, translating from English to French. "
    previous: list[Translate] = Field(default_factory=list)

    def chat(self, prompt: str) -> Reply:
        return Reply(self._stream(prompt))

    def _stream(self, prompt) -> Iterator:
        max_traslations = 3
        if len(self.previous) >= max_traslations:
            yield f"Sorry. French lesson over.\nYour {max_traslations} translations:\n"
            for resp in self.previous:
                assert isinstance(resp, Translate)
                yield f"* {resp.english} => {resp.french}\n"
        else:
            down_prompt = Markdown()
            down_prompt += f"You are a translator, translating from English to French. "
            down_prompt += f"English Text: {prompt}"
            down_prompt += reply_in_json(Translate)

            translated: Translate = self.ask_llm(down_prompt, format=Translate)

            self.previous.append(translated)
            remaining = max_traslations - len(self.previous)

            yield f"{translated.english} in French is {translated.french}\n"
            yield "\n"
            yield f"{remaining} translation(s) left."


# In a real example, validate and retry would be added to provide robustness.

session = connect_chatbot(FrenchAgent(model=connect("mistral"))) | echo()

print("--[ User-facing conversation ]------")
session = session.chat("Three blind mice")
session = session.chat("Such is life")
session = session.chat("All roads lead to Rome")
session = session.chat("The quick brown fox")

session = connect_chatbot(FrenchAgent(model=connect("mistral") | echo()))

print("--[ LLM-facing conversation ]------")
session = session.chat("Three blind mice")
session = session.chat("Such is life")
session = session.chat("All roads lead to Rome")
session = session.chat("The quick brown fox")

import json
from dataclasses import dataclass, field, replace

from pydantic import BaseModel, Field

from haverscript import *
from haverscript.languagemodel import *
from haverscript.middleware import *


class Translate(BaseModel):
    english: str
    french: str


class ExampleMetaModel(MetaModel):
    previous: list = Field(default_factory=list)

    def chat(self, prompt, next: LanguageModel) -> Reply:

        if len(self.previous) >= 3:
            response = Reply("Sorry. French lesson over.\nYour 3 translations:\n")
            for resp in self.previous:
                response += Reply(f"* {resp}\n")

            return response

        request = Request(
            contexture=Contexture(model="mistral"),
            prompt=f"You are a translator, translating from English to French. "
            f'Give your answer as a JSON record with two fields, "english", and "french". '
            f'The "english" field should contain the original English text, and only the original text.'
            f'The "french" field should contain the translated French text, and only the translated text.'
            f"\n\nEnglish Text: {prompt}",
            format=Translate.model_json_schema(),
        )
        response = next.ask(request)
        translated = response.parse(Translate)

        self.previous.append(translated)
        remaining = 3 - len(self.previous)

        return Reply(
            f"{translated.english} in French is {translated.french}\n{remaining} translation(s) left."
        )


# In a real example, validate and retry would be added to provide robustness.
model = connect("mistral").options(seed=12345) | meta(ExampleMetaModel) | echo()


print("--[ User-facing conversation ]------")
session = model.chat("Three blind mice")
session = session.chat("Such is life")
session = session.chat("All roads lead to Rome")
session = session.chat("The quick brown fox")

model = connect("mistral").options(seed=12345) | echo() | meta(ExampleMetaModel)

print("--[ LLM-facing conversation ]------")
session = model.chat("Three blind mice")
session = session.chat("Such is life")
session = session.chat("All roads lead to Rome")
session = session.chat("The quick brown fox")

from haverscript import connect, fresh, valid_json
import json

model = connect("mistral").echo()
session = model.chat("In one sentence, why is the sky blue?").check(
    fresh
)  # will ignore cache if cache enabled
session = session.chat("Rewrite the above sentence in the style of Yoda")
session = session.chat("How many questions did I ask?")

# only accept short replies
session = session.check(lambda pred: len(pred.reply) < 100)
print(f"{len(session.reply)} characters")

# turn off echo
model = model.echo(False)

colors = (
    model.json()
    .chat(
        "Return a map from primary colors to their hex codes. The map should be called colors. Reply using JSON, and only JSON."
    )
    .json_value()  # read the JSON value (retries until valid JSON)
)

print(json.dumps(colors, indent=2))

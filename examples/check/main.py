import haverscript as hs
import json

model = hs.model("mistral").echo()
session = model.chat("In one sentence, why is the sky blue?").check(
    hs.fresh
)  # will ignore cache if cache enabled
session = session.chat("Rewrite the above sentence in the style of Yoda")
session = session.chat("How many questions did I ask?")

# only accept short replies
session = session.check(lambda pred: len(pred.reply) < 100, limit=10)
print(f"{len(session.reply)} characters")

# turn off echo
model = model.echo(False)

colors = json.loads(
    model.json()
    .chat(
        "Return a map from primary colors to their hex codes. The map should be called colors. Reply using JSON, and only JSON."
    )
    .check(hs.valid_json)
    .reply
)

print(json.dumps(colors, indent=2))

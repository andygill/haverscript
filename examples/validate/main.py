from haverscript import (
    connect,
    Model,
    stop_after_attempt,
    echo,
    validate,
    retry,
)

model: Model = connect("mistral")
session = model | echo()
session = session.chat("In one sentence, why is the sky blue?")
session = session.chat("Rewrite the above sentence in the style of Yoda")

session = session.chat(
    "How many questions did I ask? Give a one sentence reply.",
    middleware=validate(lambda reply: len(reply) <= 100)  # Ensure the reply is short
    | retry(stop=stop_after_attempt(10)),
)

print(f"{len(session.reply)} characters in reply")

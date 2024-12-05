# required HaverScript 0.2
from haverscript import connect, stop_after_attempt

model = connect("mistral").echo()

best = (
    model.validate(  # Ensure the reply is three words or fewer
        lambda response: len(response.split()) <= 3
    )
    .retry(stop=stop_after_attempt(10))
    .chat(
        "Name the best basketball player. Only name one player and do not give commentary."
    )
)

model.chat(
    f"Someone told me that {best} is the best basketball player. Do you agree, and why?"
)

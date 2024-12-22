from haverscript import connect, stop_after_attempt, echo, validate, retry


def small(reply):
    # Ensure the reply is three words or fewer
    return len(reply.split()) <= 3


model = connect("mistral") | echo()

best = model.chat(
    "Name the best basketball player. Only name one player and do not give commentary.",
    middleware=validate(small) | retry(stop=stop_after_attempt(10)),
)
model.chat(
    f"Someone told me that {best} is the best basketball player. Do you agree, and why?"
)

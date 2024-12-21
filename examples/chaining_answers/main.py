from haverscript import connect, stop_after_attempt, echo, validate, retry

model = (
    connect("mistral")
    | echo()
    | validate(  # Ensure the reply is three words or fewer
        lambda response: len(response.split()) <= 3
    )
    | retry(stop=stop_after_attempt(10))
)

best = model.chat(
    "Name the best basketball player. Only name one player and do not give commentary."
)

model = connect("mistral") | echo()

model.chat(
    f"Someone told me that {best} is the best basketball player. Do you agree, and why?"
)

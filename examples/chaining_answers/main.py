from haverscript import connect

model = connect("mistral").echo()

best = model.chat(
    "Name the best basketball player. Only name one player and do not give commentary."
).check(  # Ensure the reply is three words or fewer
    lambda response: len(response.reply.split()) <= 3
)

model.chat(
    f"Someone told me that {best} is the best basketball player. Do you agree, and why?"
)

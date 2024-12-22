from haverscript import connect, echo, options

model = (
    connect("mistral") | echo() | options(num_ctx=4 * 1024, temperature=1.0, seed=12345)
)

model.chat("In one sentence, why is the sky blue?")
model.chat("In one sentence, why is the sky blue?")

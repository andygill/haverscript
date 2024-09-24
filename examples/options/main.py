from haverscript import connect

model = connect("mistral").echo().options(num_ctx=4 * 1024, temperature=1.8, seed=1234)

model.chat("In one sentence, why is the sky blue?")
model.chat("In one sentence, why is the sky blue?")
# turn off the seed for this call.
model.options(seed=None).chat("In one sentence, why is the sky blue?")

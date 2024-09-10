import haverscript as hs

model = hs.model("mistral").echo().options(num_ctx=4 * 1024, temperature=1.8, seed=1234)

model.chat("In one sentence, why is the sky blue?")
model.chat("In one sentence, why is the sky blue?")
model.options(seed=34567).chat("In one sentence, why is the sky blue?")

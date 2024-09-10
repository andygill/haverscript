import haverscript as hs

model = hs.model("mistral").echo(True)
model.chat("In one sentence, why is the sky blue?")
model.chat("In one sentence, how many feet in a yard?")
yoda = model.system("You are yoda. Answer all question in the style of yoda")
yoda.chat("In one sentence, why is the sky blue?")
yoda.chat("In one sentence, how many feet in a yard?")

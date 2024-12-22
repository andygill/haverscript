from haverscript import connect, echo

model = connect("mistral") | echo()
model.chat("In one sentence, why is the sky blue?")
model.chat("In one sentence, how many inches in a feet?")
# Set system prompt to Yoda's style
yoda = model.system("You are yoda. Answer all question in the style of yoda.")
yoda.chat("In one sentence, why is the sky blue?")
yoda.chat("In one sentence, how many inches in a feet?")

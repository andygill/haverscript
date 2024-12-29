from haverscript import echo
from haverscript.together import connect

session = connect("meta-llama/Meta-Llama-3-8B-Instruct-Lite") | echo()

session = session.chat("Write a short sentence on the history of Scotland.")
session = session.chat("Write 500 words on the history of Scotland.")
session = session.chat("Who was the most significant individual in Scottish history?")

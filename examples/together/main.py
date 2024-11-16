from haverscript import connect
from haverscript.together import Together
from tenacity import stop_after_attempt, wait_fixed

model = "meta-llama/Meta-Llama-3-8B-Instruct-Lite"
session = (
    connect(model, service=Together)
    .retry_policy(stop=stop_after_attempt(5), wait=wait_fixed(2))
    .echo()
)
session = session.chat("Write a short sentence on the history of Scotland.")
session = session.chat("Write 100 words on the history of Scotland.")
session = session.chat("Write 1000 words on the history of Scotland.")

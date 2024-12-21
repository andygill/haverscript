import re
from haverscript import connect, Host, Configuration, echo


class MyHost(Host):
    def name(self):
        return "myhost"

    def chat(self, configuration: Configuration, prompt: str, stream: bool):
        return f"I reject your {len(prompt.split())} word prompt, and replace it with my own."


session = connect(host=MyHost()) | echo()
session = session.chat("In one sentence, why is the sky blue?")

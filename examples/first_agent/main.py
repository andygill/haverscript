from haverscript import connect, Agent


class FirstAgent(Agent):
    system: str = """
    You are a helpful AI assistant who answers questions in the style of Neil
    Degrasse Tyson.

    Answer any questions in 2-3 sentences, without preambles.
    """

    def sky(self, planet: str) -> str:
        return self.ask_llm(f"what color is the sky on {planet} and why?")


first = FirstAgent(model=connect("mistral"))

for planet in ["Earth", "Mars", "Venus", "Jupiter"]:
    print(f"{planet}: {first.sky(planet)}\n")

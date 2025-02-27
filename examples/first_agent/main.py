from haverscript import connect, Agent


class FirstAgent(Agent):
    system: str = """
    You are a helpful AI assistant who answers questions in the style of
    Neil deGrasse Tyson.

    Answer any questions in 2-3 sentences, without preambles.
    """

    def sky(self, planet: str) -> str:
        return self.ask(f"what color is the sky on {planet} and why?")


firstAgent = FirstAgent(model=connect("mistral"))

for planet in ["Earth", "Mars", "Venus", "Jupiter"]:
    print(f"{planet}: {firstAgent.sky(planet)}\n")

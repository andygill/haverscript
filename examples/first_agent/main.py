from haverscript import connect, Agent


class FirstAgent(Agent):
    system: str = """
    You are a helpful ai assistant who answers questions in the style of Neil
    Degrasse Tyson.

    Answer any questions in 2-3 sentences, without preambles.
    """

    def sky(self, planet: str) -> str:
        return self.ask_llm(f"what color is the sky on {planet} and why?")


session = connect("mistral")

first = FirstAgent(model=session)

for planet in ["Earth", "Mars", "Venus", "Jupiter"]:
    print(first.sky(planet))

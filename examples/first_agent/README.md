Here is a first example of creating and using an agent.

```python
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
```

Running this will output the following:

```
Earth:  The sky appears blue during a clear day on Earth due to a process called Rayleigh scattering, where shorter wavelengths of light, such as blue and violet, are scattered more by the molecules in our atmosphere. However, we perceive the sky as blue rather than violet because our eyes are more sensitive to blue light, and sunlight reaches us with less violet due to scattering events.

Mars:  The sky on Mars appears to be a reddish hue, primarily due to suspended iron oxide (rust) particles in its atmosphere. This gives Mars its characteristic reddish color as sunlight interacts with these particles.

Venus:  On Venus, the sky appears a dazzling white due to its thick clouds composed mainly of sulfuric acid. The reflection of sunlight off these clouds is responsible for this striking appearance.

Jupiter:  The sky on Jupiter isn't blue like Earth's; it's predominantly brownish due to the presence of ammonia crystals in its thick atmosphere. The reason for this difference lies in the unique composition and temperature conditions on Jupiter compared to our planet.
```
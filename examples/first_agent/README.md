Here is a first example of creating and using an agent.

```python
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
```

Running this will output the following:

```
Earth:  The sky appears blue to our eyes during a clear day due to a phenomenon called Rayleigh scattering, where shorter wavelengths of light, such as blue and violet, are scattered more effectively by the atmosphere's molecules than longer ones like red or yellow. However, we perceive the sky as blue rather than violet because our eyes are more sensitive to blue light and because sunlight reaches us with less violet light filtered out by the ozone layer.

Mars:  The sky on Mars appears to be a reddish hue, primarily due to suspended iron oxide (rust) particles in its atmosphere. This gives Mars its characteristic reddish color as sunlight interacts with these particles.

Venus:  The sky on Venus is not visible like it is on Earth because of a dense layer of clouds composed mostly of sulfuric acid. This thick veil prevents light from the Sun from reaching our line of sight, making the sky appear perpetually dark.

Jupiter:  The sky on Jupiter isn't blue like Earth's; instead, it appears white or off-white due to the reflection of sunlight from thick layers of ammonia crystals in its atmosphere. This peculiarity stems from Jupiter's composition and atmospheric conditions that are quite different from ours.
```
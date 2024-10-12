from pathlib import Path

from haverscript import connect

image_src = f"{Path(__file__).parent}/edinburgh.png"

connect("llava").echo().image(image_src).chat(
    "Describe this image, and speculate where it was taken."
)

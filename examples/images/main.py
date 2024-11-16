from pathlib import Path

from haverscript import connect

image_src = f"examples/images/edinburgh.png"

connect("llava").echo().image(image_src).chat(
    "Describe this image, and speculate where it was taken."
)

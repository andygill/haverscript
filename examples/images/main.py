from pathlib import Path

from haverscript import connect

image_src = f"examples/images/edinburgh.png"

connect("llava").echo().chat(
    "Describe this image, and speculate where it was taken.", images=[image_src]
)

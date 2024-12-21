from pathlib import Path

from haverscript import connect, echo

image_src = f"examples/images/edinburgh.png"

model = connect("llava") | echo()

model.chat("Describe this image, and speculate where it was taken.", images=[image_src])

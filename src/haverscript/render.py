from .types import Prompt


def _canonical_string(string, postfix="\n"):
    """Adds a newline to a string if needed, for outputting to a file."""
    if not string:
        return string
    if not string.endswith(postfix):
        overlap_len = next(
            (i for i in range(1, len(postfix)) if string.endswith(postfix[:i])), 0
        )
        string += postfix[overlap_len:]
    return string


def render_system(system) -> str:
    return _canonical_string(system or "")


def render_interaction(context: str, prompt: Prompt, reply: str) -> str:
    assert isinstance(context, str)
    assert isinstance(
        prompt, Prompt
    ), f"expecting prompt:Prompt, found {type(prompt)}, {prompt}"
    assert isinstance(reply, str)

    context = _canonical_string(context, postfix="\n\n")

    prompt = prompt.content

    if prompt:
        prompt = "".join([f"> {line}\n" for line in prompt.splitlines()]) + "\n"
    else:
        prompt = ">\n\n"

    reply = reply or ""

    return context + prompt + _canonical_string(reply.strip())

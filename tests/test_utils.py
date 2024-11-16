import re


def remove_spinner(text):
    # This removes any spinner output
    return re.sub(r"([^\n]*\r)+", "", text)

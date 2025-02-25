# End to end tests for Haverscript.

import os
import subprocess
import sys

import pytest
from pydantic import BaseModel, ConfigDict, Field

import haverscript
import haverscript.together as together
from haverscript.together import connect
from tests.test_utils import remove_spinner

DEBUG = False


def open_as_is(filename, tmp_path):
    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()

    return content


def open_for_ollama(filename, tmp_path):
    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()

    content = "import haverscript as hs\n" + content
    changes = {
        'connect("mistral")': '(connect("mistral:v0.3") | hs.options(seed=12345))',
        'connect("llava")': '(connect("llava:v1.6") | hs.options(seed=12345))',
        'cache("cache.db")': f'cache("{tmp_path}/ollama.cache.db")',
    }
    for old_text, new_text in changes.items():
        content = content.replace(old_text, new_text)

    if DEBUG:
        print(content)

    return content


def open_for_together(filename, tmp_path):
    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()

    content = (
        "import haverscript as hs\nimport haverscript.together as together\n" + content
    )
    changes = {
        'connect("mistral")': '(together.connect("meta-llama/Meta-Llama-3-8B-Instruct-Lite") | hs.options(seed=12345))',
        'cache("cache.db")': f'cache("{tmp_path}/cache.together.db")',
        "connect(model)": "(connect(model) |  hs.options(seed=12345))",
    }
    for old_text, new_text in changes.items():
        content = content.replace(old_text, new_text)

    if DEBUG:
        print("# TOGETHER")
        print(content)

    return content


def run_example(example, tmp_path, open_me, args=[]) -> str:
    if DEBUG:
        print("run_example", example, tmp_path, open_me, args)
    filename = tmp_path / os.path.basename(example)

    content = open_me(example, tmp_path)

    # Write the modified content to the output file
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)

    result = subprocess.run(
        [sys.executable, filename] + args,
        capture_output=True,
        text=False,
    )

    # check everything ran okay
    if result.stderr:
        print(result.stderr.decode("utf-8"))

    if DEBUG:
        print(result.stdout.decode("utf-8"))

    assert not result.stderr, "{result.stderr}"

    return remove_spinner(result.stdout.decode("utf-8"))


class Similarity(BaseModel):
    reasons: str
    similarity: float


def check_together(f1, f2):
    with open(f1) as f:
        s1 = f.read()
    with open(f2) as f:
        s2 = f.read()

    s1 = f'"""{s1}"""'
    s2 = f'"""{s2}"""'

    prompt = f"""
Here are two articles inside triple quotes. 
Consider if the articles cover similar topics and provide similar information,
and are of similar length.
Give you answer as using JSON, and only JSON. There are two fields:
 * "similarity": int # a score between 0 and 1 of how similar the articles are
 * "reasons": str # a one sentence summary

---
article 1 : {s1}
---
article 2 : {s2}
"""

    response = connect("meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo").chat(
        prompt, middleware=haverscript.format(Similarity)
    )

    similarity = Similarity.model_validate(response.value)

    if similarity.similarity < 0.5:
        raise AssertionError()


def run_examples(
    tmp_path,
    file_regression,
    src,
    as_is: bool = False,
    ollama: bool = True,
    together: bool = True,
    arg: str | None = None,
):
    suffix = ".txt"
    args = []
    if arg:
        suffix = f".{arg}.txt"
        args = [arg]
    # Check the given example actually compiles and runs
    example = run_example(
        src,
        tmp_path,
        open_as_is,
        args=args,
    )

    if as_is:
        file_regression.check(
            example,
            extension=f".as_is{suffix}",
        )
    # Check vs golden output for ollama
    if ollama:
        file_regression.check(
            run_example(
                src,
                tmp_path,
                open_for_ollama,
                args=args,
            ),
            extension=suffix,
        )
    # Check vs golden output for together
    if together:
        file_regression.check(
            run_example(
                src,
                tmp_path,
                open_for_together,
                args=args,
            ),
            check_fn=check_together,
            extension=f".together{suffix}",
        )


def test_first_example(tmp_path, file_regression):
    run_examples(tmp_path, file_regression, "examples/first_example/main.py")


def test_chaining_answers(tmp_path, file_regression):
    run_examples(tmp_path, file_regression, "examples/chaining_answers/main.py")


def test_tree_of_calls(tmp_path, file_regression):
    run_examples(tmp_path, file_regression, "examples/tree_of_calls/main.py")


def test_images(tmp_path, file_regression):
    run_examples(tmp_path, file_regression, "examples/images/main.py", together=False)


def test_cache(tmp_path, file_regression):
    run_examples(tmp_path, file_regression, "examples/cache/main.py", arg="2")
    run_examples(tmp_path, file_regression, "examples/cache/main.py", arg="3")


def test_together(tmp_path, file_regression):
    run_examples(tmp_path, file_regression, "examples/together/main.py", ollama=False)


def test_options(tmp_path, file_regression):
    run_examples(tmp_path, file_regression, "examples/options/main.py")


def test_format(tmp_path, file_regression):
    run_examples(tmp_path, file_regression, "examples/format/main.py")


def test_chatbot(tmp_path, file_regression):
    run_examples(tmp_path, file_regression, "examples/chatbot/main.py")


def test_custom_service(tmp_path, file_regression):
    run_examples(
        tmp_path,
        file_regression,
        "examples/custom_service/main.py",
        together=False,
        ollama=False,
        as_is=True,
    )


def test_list():
    models = haverscript.connect().list()
    assert isinstance(models, list)
    assert "mistral:v0.3" in models
    assert "llava:v1.6" in models

    models = together.connect().list()
    assert isinstance(models, list)
    assert "meta-llama/Meta-Llama-3-8B-Instruct-Lite" in models

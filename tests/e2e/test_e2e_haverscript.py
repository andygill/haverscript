# End to end tests for HaverScript.

import subprocess
import sys
import os
from pydantic import BaseModel, ConfigDict, Field

import pytest
import haverscript
from haverscript.together import connect

from tests.test_utils import remove_spinner


def run_example(example, tmp_path, header, changes, args=[]) -> str:
    filename = tmp_path / os.path.basename(example)

    with open(example, "r", encoding="utf-8") as f:
        content = f.read()

    # Apply the diffs
    if header:
        content = header.strip() + "\n" + content
    for old_text, new_text in changes.items():
        content = content.replace(old_text, new_text)

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
        prompt, format=Similarity.model_json_schema()
    )

    similarity = Similarity.model_validate(response.value)

    if similarity.similarity < 0.5:
        raise AssertionError()


def test_first_example(tmp_path, file_regression):
    file_regression.check(
        run_example(
            "examples/first_example/main.py",
            tmp_path,
            "import haverscript as hs",
            {
                '("mistral")': '("mistral:v0.3") | hs.options(seed=12345)',
            },
        ),
        extension=".txt",
    )


def test_first_example_together(tmp_path, file_regression):
    file_regression.check(
        run_example(
            "examples/first_example/main.py",
            tmp_path,
            "import haverscript as hs\nimport haverscript.together as together",
            {
                '("mistral")': '("meta-llama/Meta-Llama-3-8B-Instruct-Lite") | hs.retry(stop=hs.stop_after_attempt(5), wait=hs.wait_fixed(2)) | hs.options(seed=12345)',
                "connect(": "together.connect(",
            },
        ),
        extension=".txt",
        check_fn=check_together,
    )


def test_together(tmp_path, file_regression):
    file_regression.check(
        run_example(
            "examples/together/main.py",
            tmp_path,
            "import haverscript as hs\n",
            {"connect(model)": "connect(model) | hs.options(seed=12345)"},
        ),
        extension=".txt",
        check_fn=check_together,
    )


def test_tree_of_calls(tmp_path, file_regression):
    file_regression.check(
        run_example(
            "examples/tree_of_calls/main.py",
            tmp_path,
            "",
            {'("mistral")': '("mistral:v0.3").options(seed=23456)'},
        ),
        extension=".txt",
    )


def test_tree_of_calls_together(tmp_path, file_regression):
    file_regression.check(
        run_example(
            "examples/tree_of_calls/main.py",
            tmp_path,
            "import haverscript as hs\nimport haverscript.together as together",
            {
                '("mistral")': '("meta-llama/Meta-Llama-3-8B-Instruct-Lite") | hs.options(seed=12345)',
                "connect(": "together.connect(",
            },
        ),
        extension=".txt",
    )


def test_chaining_answers(tmp_path, file_regression):
    file_regression.check(
        run_example(
            "examples/chaining_answers/main.py",
            tmp_path,
            "",
            {'("mistral")': '("mistral:v0.3").options(seed=12345)'},
        ),
        extension=".txt",
    )


def test_options(tmp_path, file_regression):
    file_regression.check(
        run_example(
            "examples/options/main.py",
            tmp_path,
            "",
            {"(seed=None)": "(seed=56789)"},
        ),
        extension=".txt",
    )


def test_cache(tmp_path, file_regression):
    env = {
        '("mistral")': '("mistral:v0.3").options(seed=12345)',
        'cache("cache.db")': f'cache("{tmp_path}/cache.db")',
    }
    file_regression.check(
        run_example(
            "examples/cache/main.py",
            tmp_path,
            "",
            env,
            args=["2"],
        ),
        extension=".2.txt",
    )
    file_regression.check(
        run_example(
            "examples/cache/main.py",
            tmp_path,
            "",
            env,
            args=["3"],
        ),
        extension=".3.txt",
    )


def test_validate(tmp_path, file_regression):
    file_regression.check(
        run_example(
            "examples/validate/main.py",
            tmp_path,
            "",
            {
                '("mistral")': '("mistral:v0.3")',
                'blue?")': 'blue?", middleware=haverscript.options(seed=12345))',
                'Yoda")': 'Yoda", middleware=haverscript.options(seed=12345))',
                "retry(stop=stop_after_attempt(10))": "retry(stop=stop_after_attempt(10)) | haverscript.options(seed=12345)",
                "from haverscript": "import haverscript\nfrom haverscript",
            },
        ),
        extension=".txt",
    )


def test_images(tmp_path, file_regression):
    file_regression.check(
        run_example(
            "examples/images/main.py",
            tmp_path,
            "",
            {'("llava")': '("llava:v1.6").options(seed=12345)'},
        ),
        extension=".txt",
    )


def test_list():
    models = haverscript.connect().list()
    assert isinstance(models, list)
    assert "mistral:v0.3" in models
    assert "llava:v1.6" in models


def test_meta(tmp_path, file_regression):
    file_regression.check(
        run_example(
            "examples/meta_model/main.py",
            tmp_path,
            "import haverscript as hs",
            {'("mistral")': '("mistral:v0.3") | hs.options(seed=12345)'},
        ),
        extension=".txt",
    )

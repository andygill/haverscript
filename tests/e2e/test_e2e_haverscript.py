# End to end tests for HaverScript.

import subprocess
import sys
import os

import pytest
import haverscript

from tests.test_utils import remove_spinner


def run_example(example, tmp_path, changes, args=[]):
    filename = tmp_path / os.path.basename(example)

    with open(example, "r", encoding="utf-8") as f:
        content = f.read()

    # Apply the diffs
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
    assert not result.stderr, "{result.stderr}"

    return remove_spinner(result.stdout.decode("utf-8"))


def test_first_example(tmp_path, file_regression):
    file_regression.check(
        run_example(
            "examples/first_example/main.py",
            tmp_path,
            {
                '("mistral")': '("mistral:v0.3") | haverscript.options(seed=12345)',
                "from haverscript": "import haverscript\nfrom haverscript",
            },
        ),
        extension=".txt",
    )


def test_first_example_together(tmp_path, file_regression):
    lines = len(
        run_example(
            "examples/first_example/main.py",
            tmp_path,
            {
                "session = connect": """
import haverscript
from haverscript.together import Together

session = connect""",
                '("mistral")': '("meta-llama/Meta-Llama-3-8B-Instruct-Lite", service=Together()) | haverscript.retry(stop=haverscript.stop_after_attempt(5), wait=haverscript.wait_fixed(2))',
            },
        ).splitlines()
    )
    assert lines > 10 and lines < 1000


def test_together(tmp_path, file_regression):
    lines = len(
        run_example(
            "examples/together/main.py",
            tmp_path,
            {},
        ).splitlines()
    )
    assert lines > 10 and lines < 1000


def test_tree_of_calls(tmp_path, file_regression):
    file_regression.check(
        run_example(
            "examples/tree_of_calls/main.py",
            tmp_path,
            {'("mistral")': '("mistral:v0.3").options(seed=23456)'},
        ),
        extension=".txt",
    )


def test_chaining_answers(tmp_path, file_regression):
    file_regression.check(
        run_example(
            "examples/chaining_answers/main.py",
            tmp_path,
            {'("mistral")': '("mistral:v0.3").options(seed=12345)'},
        ),
        extension=".txt",
    )


def test_options(tmp_path, file_regression):
    file_regression.check(
        run_example(
            "examples/options/main.py",
            tmp_path,
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
            env,
            args=["2"],
        ),
        extension=".2.txt",
    )
    file_regression.check(
        run_example(
            "examples/cache/main.py",
            tmp_path,
            env,
            args=["3"],
        ),
        extension=".3.txt",
    )


@pytest.mark.xfail
def test_check(tmp_path, file_regression):
    file_regression.check(
        run_example(
            "examples/check/main.py",
            tmp_path,
            {'("mistral")': '("mistral:v0.3").options(seed=12345)'},
        ),
        extension=".txt",
    )


def test_images(tmp_path, file_regression):
    file_regression.check(
        run_example(
            "examples/images/main.py",
            tmp_path,
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
            {'("mistral")': '("mistral:v0.3") | options(seed=12345)'},
        ),
        extension=".txt",
    )

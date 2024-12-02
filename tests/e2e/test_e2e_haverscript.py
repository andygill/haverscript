# End to end tests for HaverScript.

import subprocess
import sys
import os

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
            {'("mistral")': '("mistral:v0.3").options(seed=12345)'},
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
from haverscript.together import Together
from tenacity import stop_after_attempt, wait_fixed

session = connect""",
                '("mistral")': '("meta-llama/Meta-Llama-3-8B-Instruct-Lite", service=Together()).retry_policy(stop=stop_after_attempt(5), wait=wait_fixed(2))',
            },
        ).splitlines()
    )
    assert lines > 10 and lines < 1000


def test_together(tmp_path, file_regression):
    lines = len(
        run_example(
            "examples/together/main.py",
            tmp_path,
            {"connect(model, service=Together)": "connect(model, service=Together())"},
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

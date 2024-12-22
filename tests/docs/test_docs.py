from tests.test_utils import Content


def test_docs_first_example():
    assert (
        Content("examples/first_example/main.py")
        == Content("examples/first_example/README.md")[4:11]
    )
    assert (
        Content("tests/e2e/test_e2e_haverscript/test_first_example.txt")[2:]
        == Content("examples/first_example/README.md")[16:40]
    )


def test_docs_tree_of_calls():
    assert (
        Content("examples/tree_of_calls/main.py")
        == Content("examples/tree_of_calls/README.md")[5:14]
    )
    assert (
        Content("tests/e2e/test_e2e_haverscript/test_tree_of_calls.txt")[2:]
        == Content("examples/tree_of_calls/README.md")[19:37]
    )


def test_docs_chaining_answers():
    assert (
        Content("examples/chaining_answers/main.py")
        == Content("examples/chaining_answers/README.md")[5:22]
    )
    assert (
        Content("tests/e2e/test_e2e_haverscript/test_chaining_answers.txt")[2:]
        == Content("examples/chaining_answers/README.md")[26:52]
    )

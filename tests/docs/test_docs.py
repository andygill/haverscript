from tests.test_utils import Content


def test_docs_first_example():
    assert (
        Content("examples/first_example/main.py")
        == Content("examples/first_example/README.md")[4:10]
    )
    assert (
        Content("tests/e2e/test_e2e_haverscript/test_first_example.txt")[2:]
        == Content("examples/first_example/README.md")[15:39]
    )

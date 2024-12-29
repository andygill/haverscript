from tests.test_utils import Content


class README:
    def __init__(self, doc):
        self.doc = doc

    def __call__(self, ref, start, size, skip=0):
        ref_content = Content(ref)
        if skip:
            # line numbers start 1 (not 0)
            ref_content = ref_content[skip + 1 :]
        assert ref_content == Content(self.doc)[start : start + size]


def test_docs_first_example():
    readme = README("examples/first_example/README.md")
    readme("examples/first_example/main.py", 4, 8)
    readme("tests/e2e/test_e2e_haverscript/test_first_example.txt", 17, 20, skip=1)


def test_docs_tree_of_calls():
    readme = README("examples/tree_of_calls/README.md")
    readme("examples/tree_of_calls/main.py", 5, 9)
    readme("tests/e2e/test_e2e_haverscript/test_tree_of_calls.txt", 19, 18, skip=1)


def test_docs_chaining_answers():
    readme = README("examples/chaining_answers/README.md")
    readme("examples/chaining_answers/main.py", 5, 17)
    readme("tests/e2e/test_e2e_haverscript/test_chaining_answers.txt", 26, 26, skip=1)


def test_readme():
    readme = README("README.md")

    readme("examples/first_example/main.py", 20, 8)
    readme("tests/e2e/test_e2e_haverscript/test_first_example.txt", 33, 20, skip=1)
    assert (
        Content("docs/MIDDLEWARE.md")[50 : 50 + 14]
        == Content("README.md")[284 : 284 + 14]
    )
    readme("examples/together/main.py", 331, 8)

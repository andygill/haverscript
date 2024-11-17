import difflib


class Content:
    def __init__(self, filename: str, content=None, start_line=0, end_line=None):
        if content is not None:
            # Content is provided directly (from slicing)
            self.content = content
            self.start_line = start_line
            self.end_line = (
                end_line if end_line is not None else len(content) + start_line
            )
            self.filename = f"{filename}[{self.start_line}:{self.end_line}]"
        else:
            # Read content from file
            self.filename = filename
            with open(filename) as f:
                self.content = f.read().splitlines()
            self.start_line = 0
            self.end_line = len(self.content)

    def __getitem__(self, key):
        if isinstance(key, slice):
            # Adjust slice indices. These are line numbers (aka start at 1)
            start = key.start if key.start is not None else 1
            stop = key.stop if key.stop is not None else len(self.content) + 1

            # Slice the content
            new_content = self.content[start - 1 : stop - 1]

            # Compute new start and end lines
            new_start_line = self.start_line + start
            new_end_line = self.start_line + stop

            return Content(
                filename=self.filename,
                content=new_content,
                start_line=new_start_line,
                end_line=new_end_line,
            )
        else:
            raise TypeError("Invalid argument type for indexing")

    def __eq__(self, other):
        assert isinstance(other, Content)

        if self.content != other.content:
            diff = "\n".join(
                difflib.unified_diff(
                    self.content,
                    other.content,
                    fromfile=self.filename,
                    tofile=other.filename,
                    lineterm="",
                )
            )
            assert False, f"Contents are different:\n{diff}"

        return True

    def __str__(self):
        return "\n".join(self.content)


def test_docs_first_example():
    assert (
        Content("examples/first_example/main.py")
        == Content("examples/first_example/README.md")[4:10]
    )
    assert (
        Content("tests/e2e/test_e2e_haverscript/test_first_example.txt")[2:]
        == Content("examples/first_example/README.md")[15:32]
    )

import pytest
from haverscript import connect, Model, Response, valid_json, fresh
import sys
import json
import re

test_model_name = "test-model"
test_model_host = "remote.address"


def llm(host, model, messages, options, format, extra=None):
    return json.dumps(
        {
            "host": host,
            "model": model,
            "messages": messages,
            "options": options,
            "format": format,
            "extra": extra,
        },
    )


class _TestClient:
    def __init__(self, host) -> None:
        self._host = host
        self._count = 1

    def chat(self, model, stream, messages, options, format):
        assert format == "json" or format == ""
        extra = None
        if "###" in messages[-1]["content"]:
            extra = self._count
            self._count += 1
        reply = llm(self._host, model, messages, options, format, extra)
        if stream:
            return [
                {"message": {"content": token}}
                for token in re.findall(r"\S+|\s+", reply)
            ]
        else:
            return {"message": {"content": reply}}


# inject the TestClient
ollama = sys.modules["haverscript.haverscript"].services._ollama
ollama[None] = _TestClient(None)
ollama[test_model_host] = _TestClient(test_model_host)


@pytest.fixture
def sample_model():
    return connect(test_model_name)


@pytest.fixture
def sample_remote_model():
    return connect(test_model_name, hostname=test_model_host)


def test_model(sample_model):
    check_model(sample_model, None)


def test_remote_model(sample_remote_model):
    check_model(sample_remote_model, test_model_host)


def check_model(model, host):
    assert type(model) is Model
    assert hasattr(model, "configuration")
    config = model.configuration
    assert hasattr(config, "model")
    assert config.model == test_model_name
    assert config.host == host

    context = []
    session = model
    for i in range(5):
        message = f"Message #{i}"
        context.append({"role": "user", "content": message})
        session = session.chat(message)
        assert type(session) is Response
        assert isinstance(session.reply, str)
        assert session.reply == llm(host, test_model_name, context, {}, "")
        context.append({"role": "assistant", "content": session.reply})


reply_to_hello = """
> Hello

{"host": null, "model": "test-model", "messages": [{"role": "user", "content":
"Hello"}], "options": {}, "format": "", "extra": null}
"""

reply_to_hello_48 = """
> Hello

{"host": null, "model": "test-model",
"messages": [{"role": "user", "content":
"Hello"}], "options": {}, "format": "", "extra":
null}
"""
reply_to_hello_8 = """
> Hello

{"host":
null,
"model":
"test-model",
"messages":
[{"role":
"user",
"content":
"Hello"}],
"options":
{},
"format":
"",
"extra":
null}
"""


def test_echo(sample_model, capfd):
    capfd.readouterr()
    model = sample_model.echo()
    assert type(model) is Model
    assert model.settings.echo == True
    model = sample_model.echo(True)
    assert type(model) is Model
    assert model.settings.echo == True
    model = sample_model.echo(False)
    assert type(model) is Model
    assert model.settings.echo == False

    root_resp = sample_model.chat("Hello")
    resp = root_resp.echo()
    assert type(resp) is Response
    assert resp.settings.echo == True
    resp = root_resp.echo(True)
    assert type(resp) is Response
    assert resp.settings.echo == True
    resp = root_resp.echo(False)
    assert type(resp) is Response
    assert resp.settings.echo == False

    resp = sample_model.chat("Hello")
    assert capfd.readouterr().out == ""

    resp = sample_model.echo(False).chat("Hello")
    assert capfd.readouterr().out == ""

    for line in reply_to_hello.splitlines():
        assert len(line) <= 78

    resp = sample_model.echo(True).chat("Hello")
    assert capfd.readouterr().out == reply_to_hello

    resp = sample_model.echo().chat("Hello")
    assert capfd.readouterr().out == reply_to_hello

    for line in reply_to_hello_48.splitlines():
        assert len(line) <= 48

    resp = sample_model.echo(width=48).chat("Hello")
    assert capfd.readouterr().out == reply_to_hello_48

    for line in reply_to_hello_8.splitlines():
        assert len(line) <= 8 or " " not in line

    resp = sample_model.echo(width=8).chat("Hello")
    assert capfd.readouterr().out == reply_to_hello_8


def test_outdent(sample_model):
    messages = [
        "Hello",
        """
    Hello

    World
    """,
    ]
    model = sample_model
    for ix, model in enumerate(
        [
            sample_model,
            sample_model.outdent(True),
            sample_model.outdent(False),
        ]
    ):
        for message in messages:
            reply = model.chat(message)
            actual_message = message
            if ix != 2:
                actual_message = "\n".join(
                    [line.lstrip() for line in actual_message.splitlines()]
                ).strip()
            context = [{"role": "user", "content": actual_message}]
            assert reply.reply == llm(None, test_model_name, context, {}, "")


def test_options(sample_model):
    reply = sample_model.options(seed=12345).chat("")
    context = [{"role": "user", "content": ""}]
    assert reply.reply == llm(None, test_model_name, context, {"seed": 12345}, "")


def test_system(sample_model):
    reply = sample_model.system("woof!").chat("")
    context = [{"role": "system", "content": "woof!"}, {"role": "user", "content": ""}]
    assert reply.reply == llm(None, test_model_name, context, {}, "")


def test_json(sample_model):
    for ix, model in enumerate(
        [
            sample_model,
            sample_model.json(True),
            sample_model.json(False),
        ]
    ):
        reply = model.chat("")
        context = [{"role": "user", "content": ""}]
        assert reply.reply == llm(
            None, test_model_name, context, {}, "json" if ix == 1 else ""
        )


def test_fresh(sample_model):
    reply = sample_model.chat("Hello")
    assert reply.fresh == True


def test_cache(sample_model, tmp_path):
    temp_file = tmp_path / "cache.db"
    model = sample_model.cache(temp_file)
    reply = model.chat("Hello")
    assert reply.fresh == True
    reply = model.chat("Hello")
    assert reply.fresh == False


def test_check(sample_model):
    # simple check
    print(
        repr(
            sample_model.chat("Squirrel").check(lambda reply: "Squirrel" in reply.reply)
        )
    )

    # failing check
    with pytest.raises(RuntimeError, match="exceeded the count limit"):
        sample_model.chat("Squirrel").check(lambda reply: "Squirrel" not in reply.reply)

    # chaining checks
    sample_model.chat("Squirrel Haggis").check(
        lambda reply: "Squirrel" in reply.reply
    ).check(  # add on check for Haggis
        lambda reply: "Haggis" in reply.reply
    ).check(  # check for freshness
        fresh
    ).check(  # check output is valid JSON (the test stub used JSON for output)
        valid_json
    )


def test_check_chaining(sample_model):
    reply = (
        sample_model.chat("###")
        .check(valid_json)
        .check(lambda r: json.loads(r.reply)["extra"] in [5, 8])
        .check(lambda r: json.loads(r.reply)["extra"] > 6)
    )

    assert json.loads(reply.reply)["extra"] == 8

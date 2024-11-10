from pathlib import Path
import pytest
import sys
import json
import re
import sys

import pytest

from haverscript import (
    Configuration,
    Echo,
    ServiceProvider,
    Model,
    Response,
    connect,
    fresh,
    valid_json,
    list_models,
)

# Note that these tests break the haverscript API at points specifically
# for testing purposes.
#
# Specifically:
#   * We fake the LLM (so tests run without needing ollama).
#   * We override some fields when doing equality in the cache testing.

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

    def _streaming(self, reply):
        yield from [
            {"message": {"content": token}, "done": False}
            for token in re.findall(r"\S+|\s+", reply)
        ]

    def chat(self, model, stream, messages, options, format):
        assert format == "json" or format == ""
        extra = None
        if "###" in messages[-1]["content"]:
            extra = self._count
            self._count += 1
        reply = llm(self._host, model, messages, options, format, extra)
        if stream:
            return self._streaming(reply)
        else:
            return {
                "message": {"content": reply},
                "total_duration": 100,
                "load_duration": 101,
                "prompt_eval_count": 102,
                "prompt_eval_duration": 103,
                "eval_count": 104,
                "eval_duration": 105,
            }

    def list(self):
        return ["A", "B", "C", "D"]


# inject the TestClient
models = sys.modules["haverscript.haverscript"].services._model_providers
models["ollama@local"] = sys.modules["haverscript.haverscript"].Ollama(None)
models["ollama@local"].client = _TestClient(None)
models[f"ollama@{test_model_host}"] = sys.modules["haverscript.haverscript"].Ollama(
    test_model_host
)
models[f"ollama@{test_model_host}"].client = _TestClient(test_model_host)


@pytest.fixture
def sample_model():
    return connect(test_model_name)


@pytest.fixture
def sample_remote_model():
    return connect(test_model_name, hostname=test_model_host)


def test_model(sample_model):
    check_model(sample_model, None, None)


def test_model_and_system(sample_model):
    check_model(sample_model.system("system"), None, "system")


def test_remote_model(sample_remote_model):
    check_model(sample_remote_model, test_model_host, None)


def test_remote_model_and_system(sample_remote_model):
    check_model(sample_remote_model.system("system"), test_model_host, "system")


def check_model(model, host, system):
    assert type(model) is Model
    assert hasattr(model, "configuration")
    config = model.configuration
    assert hasattr(config, "model")
    assert config.model == test_model_name
    assert config.service == "ollama@" + (host or "local")

    render = ""
    if system:
        render = system + "\n"
    assert model.render() == render

    context = []
    if system:
        context.append({"role": "system", "content": system})
        render += "\n"
    session = model
    for i in range(2):
        message = f"Message #{i}"
        context.append({"role": "user", "content": message})
        render += "".join(["> " + line for line in message.splitlines()]) + "\n\n"
        session = session.chat(message)
        assert type(session) is Response
        assert isinstance(session.reply, str)
        assert session.reply == llm(host, test_model_name, context, {}, "")
        assert session.metrics.total_duration == 100
        assert session.metrics.load_duration == 101
        assert session.metrics.prompt_eval_count == 102
        assert session.metrics.prompt_eval_duration == 103
        assert session.metrics.eval_count == 104
        assert session.metrics.eval_duration == 105

        context.append({"role": "assistant", "content": session.reply})
        render += session.reply
        render += "\n"
        assert session.render() == render
        render += "\n"


class UserService(ServiceProvider):
    def __init__(self, hostname) -> None:
        self.hostname = hostname

    def name(self):
        return f"myhost@{self.hostname}"

    def chat(self, configuration: Configuration, prompt: str, stream: bool):
        return (
            f"I reject your {len(prompt.split())} word prompt, and replace it with my own.",
            None,
        )

    def list(self):
        return ["A", "B", "C"]


@pytest.fixture
def sample_user_model():
    return connect(service=UserService)


def test_user_model(sample_user_model):
    model = sample_user_model
    assert type(model) is Model
    assert hasattr(model, "configuration")
    config = model.configuration
    assert hasattr(config, "model")
    assert config.model is None
    assert config.service == "myhost@None"
    context = []
    session = model
    message = "Three word prompt"
    context.append({"role": "user", "content": message})
    session = session.chat(message)
    assert type(session) is Response
    assert isinstance(session.reply, str)
    assert session.reply == "I reject your 3 word prompt, and replace it with my own."


def test_list_models():
    assert list_models(service=UserService) == ["A", "B", "C"]


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


def remove_spinner(text):
    # This removes any spinner output
    return re.sub(r"([^\n]*\r)+", "", text)


def test_echo(sample_model, capfd):
    capfd.readouterr()
    model = sample_model.echo()
    assert type(model) is Model
    assert model.settings.echo == Echo(width=78)
    model = sample_model.echo(True)
    assert type(model) is Model
    assert model.settings.echo == Echo(width=78)
    model = sample_model.echo(True, width=99)
    assert type(model) is Model
    assert model.settings.echo == Echo(width=99)
    model = sample_model.echo(False)
    assert type(model) is Model
    assert model.settings.echo is None

    root_resp = sample_model.chat("Hello")
    resp = root_resp.echo()
    assert type(resp) is Response
    assert resp.settings.echo == Echo(width=78)
    assert resp.metrics.total_duration == 100
    assert resp.metrics.load_duration == 101
    assert resp.metrics.prompt_eval_count == 102
    assert resp.metrics.prompt_eval_duration == 103
    assert resp.metrics.eval_count == 104
    assert resp.metrics.eval_duration == 105

    resp = root_resp.echo(True)
    assert type(resp) is Response
    assert resp.settings.echo == Echo(width=78)
    resp = root_resp.echo(False)
    assert type(resp) is Response
    assert resp.settings.echo is None

    resp = sample_model.chat("Hello")
    assert capfd.readouterr().out == ""

    resp = sample_model.echo(False).chat("Hello")
    assert capfd.readouterr().out == ""

    for line in reply_to_hello.splitlines():
        assert len(line) <= 78

    resp = sample_model.echo(True).chat("Hello")
    assert remove_spinner(capfd.readouterr().out) == reply_to_hello

    resp = sample_model.echo().chat("Hello")
    assert remove_spinner(capfd.readouterr().out) == reply_to_hello

    for line in reply_to_hello_48.splitlines():
        assert len(line) <= 48

    resp = sample_model.echo(width=48).chat("Hello")
    assert remove_spinner(capfd.readouterr().out) == reply_to_hello_48

    for line in reply_to_hello_8.splitlines():
        assert len(line) <= 8 or " " not in line

    resp = sample_model.echo(width=8).chat("Hello")
    assert remove_spinner(capfd.readouterr().out) == reply_to_hello_8


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
        if ix == 1:
            assert reply.value is not None
            assert reply.value == json.loads(
                llm(None, test_model_name, context, {}, "json" if ix == 1 else "")
            )


def test_fresh(sample_model):
    reply = sample_model.chat("Hello")
    assert reply.fresh == True


def test_cache(sample_model, tmp_path):
    temp_file = tmp_path / "cache.db"
    model = sample_model.cache(temp_file)
    assert len(model.children("Hello")) == 0
    assert len(model.children()) == 0

    reply = model.chat("Hello")
    assert reply.fresh == True
    assert len(model.children("Hello")) == 1
    assert len(model.children()) == 1
    context = [{"role": "user", "content": "Hello"}]
    assert model.children("Hello")[0] == reply.copy(fresh=False, metrics=None)

    reply = model.chat("Hello")
    assert reply.fresh == False
    assert len(model.children("Hello")) == 1
    assert len(model.children()) == 1
    assert model.children("Hello")[0] == reply

    reply = model.chat("World")
    assert reply.fresh == True
    assert len(model.children("World")) == 1
    assert len(model.children()) == 2
    assert model.children("World")[0] == reply.copy(fresh=False, metrics=None)

    reply = model.chat("World")
    assert reply.fresh == False
    assert len(model.children("World")) == 1
    assert len(model.children()) == 2
    assert model.children("World")[0] == reply

    reply = model.chat("###")
    assert reply.fresh == True
    assert len(model.children("###")) == 1
    assert len(model.children()) == 3
    assert model.children("###")[0] == reply.copy(fresh=False, metrics=None)

    reply = model.chat("###").check(fresh)
    assert reply.fresh == True
    assert len(model.children("###")) == 2
    assert len(model.children()) == 4
    assert model.children("###")[-1] == reply.copy(
        fresh=False, _predicates=[], metrics=None
    )


def test_check(sample_model):
    # simple check
    assert repr(
        sample_model.chat("Squirrel").check(lambda reply: "Squirrel" in reply.reply)
    ).startswith("Response")

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


def test_image(sample_model):
    image_src = f"{Path(__file__).parent}/../examples/images/edinburgh.png"
    prompt = "Describe this image"
    resp = sample_model.image(image_src).chat(prompt)
    context = [{"role": "user", "content": prompt, "images": [image_src]}]
    reply = resp.reply
    assert reply == llm(None, test_model_name, context, {}, "")
    prompt2 = "Follow on question"
    resp = resp.chat(prompt2)
    context = [
        {"role": "user", "content": prompt, "images": [image_src]},
        {"role": "assistant", "content": reply},
        {"role": "user", "content": prompt2},
    ]
    assert resp.reply == llm(None, test_model_name, context, {}, "")

import json
import os
import re
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from collections import namedtuple

import pytest
from pydantic import BaseModel, Field
from tenacity import stop_after_attempt

from haverscript import (
    LanguageModel,
    LLMError,
    LLMResultError,
    Middleware,
    Model,
    Reply,
    Response,
    Service,
    ServiceProvider,
    connect,
    Markdown,
    bullets,
    header,
    text,
    quoted,
    rule,
    table,
    code,
    reply_in_json,
    tool,
)
from haverscript.cache import INTERACTION, Cache
from haverscript.types import Exchange, Request, ResponseMessage, ToolResult, ToolReply
from haverscript.middleware import *
from haverscript.utils import tool_schema
from tests.test_utils import remove_spinner

# Note that these tests break the haverscript API at points specifically
# for testing purposes.
#
# Specifically:
#   * We fake the LLM (so tests run without needing ollama).
#   * We override some fields when doing equality in the cache testing.

test_model_name = "test-model"
test_model_host = "remote.address"


class LLM(BaseModel):
    host: str | None = Field(..., description="The host of the LLM")
    model: str | None = Field(..., description="The model name")
    messages: list | None = Field(..., description="The conversation")
    options: dict | None = Field(..., description="The options")
    format: str | dict = Field(..., description="The format")
    extra: str | None = Field(..., description="Extra information")


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
        assert isinstance(reply, str)
        for token in re.findall(r"\S+|\s+", reply):
            time.sleep(0.01)
            yield {"message": {"content": token}, "done": False}

    def chat(self, model, stream, messages, options, format, tools):
        assert format == "json" or format == "" or isinstance(format, dict)
        if isinstance(format, dict):
            assert "$ref" not in json.dumps(format)
        extra = None

        assert isinstance(messages, list)
        assert len(messages) > 0
        assert isinstance(messages[-1], dict)
        assert "content" in messages[-1]
        assert isinstance(
            messages[-1]["content"], str
        ), f'expecting str, found {messages[-1]["content"]}:{type(messages[-1]["content"])}'
        if "###" in messages[-1]["content"]:
            extra = self._count
            self._count += 1

        match = re.match(r"^FAIL\((\d+)\)$", messages[-1]["content"])

        if match:
            self._count += 1
            n = int(match.group(1))
            if n != self._count:
                raise LLMError()

        reply = None

        if isinstance(format, dict):
            if format == {"type": "boolean"}:
                reply = json.dumps(True)
            elif format == {"type": "integer"}:
                reply = json.dumps(42)
            elif format == {"type": "number"}:
                reply = json.dumps(3.14)
            elif format == {"type": "string"}:
                reply = json.dumps("Hello")
            elif format == {
                "maxItems": 2,
                "minItems": 2,
                "prefixItems": [{"type": "integer"}, {"type": "boolean"}],
                "type": "array",
            }:
                reply = json.dumps([99, False])
            elif format == {"items": {"type": "integer"}, "type": "array"}:
                reply = json.dumps([1, 2, 3])
            elif format == {"items": {}, "type": "array"}:
                reply = json.dumps(["A", "B", "C"])
            elif format == {"anyOf": [{"type": "integer"}, {"type": "null"}]}:
                reply = json.dumps(101)
            elif format == {"type": "object"}:
                reply = json.dumps({"x:": "Hello"})
            elif format == {
                "$defs": {
                    "B": {
                        "properties": {
                            "payload": {"title": "Payload", "type": "integer"}
                        },
                        "required": ["payload"],
                        "title": "B",
                        "type": "object",
                    }
                },
                "properties": {
                    "payload": {
                        "properties": {
                            "payload": {"title": "Payload", "type": "integer"}
                        },
                        "required": ["payload"],
                        "title": "B",
                        "type": "object",
                    }
                },
                "required": ["payload"],
                "title": "A",
                "type": "object",
            }:
                reply = json.dumps({"payload": {"payload": 101}})
            elif format.get("title") != "LLM":
                assert False, f"unknown format: {repr(format)}"

        if tools and messages[-1]["content"] == "ask tool":
            return {
                "message": {
                    "content": "...",
                    "tool_calls": [
                        namedtuple("Tool", ["function"])(
                            namedtuple("Function", ["name", "arguments"])(
                                "foo", {"i": 99, "s": "Hello"}
                            )
                        )
                    ],
                },
                "total_duration": 100,
                "load_duration": 101,
                "prompt_eval_count": 102,
                "prompt_eval_duration": 103,
                "eval_count": 104,
                "eval_duration": 105,
                "done": True,
            }

            return {
                "message": {"content": "call tool", "tool_calls": 999},
                "done": True,
            }

        if reply is None:
            reply = llm(self._host, model, messages, options, format, extra)

        assert isinstance(reply, str)

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
                "done": True,
            }

    def list(self):
        @dataclass
        class Model:
            model: str

        return {"models": [Model(x) for x in ["A", "B", "C"]]}


# inject the TestClient
def inject():
    sys.modules["haverscript.ollama"].Ollama.client = {
        None: _TestClient(None),
        test_model_host: _TestClient(test_model_host),
    }


@pytest.fixture
def sample_model():
    inject()
    return connect(test_model_name)


@pytest.fixture
def sample_remote_model():
    inject()
    return connect(test_model_name, test_model_host)


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

        assert isinstance(session.reply, str)

        context.append({"role": "assistant", "content": session.reply})
        render += session.reply
        render += "\n"
        assert session.render() == render
        render += "\n"


class UserService(ServiceProvider):
    def ask(self, request: Request):
        return Reply(
            [
                f"I reject your {len(request.prompt.content.split())} word prompt, and replace it with my own."
            ]
        )

    def list(self):
        return ["A", "B", "C"]


@pytest.fixture
def sample_user_model():
    inject()
    return Service(UserService()) | model("A")


def test_user_model(sample_user_model):
    model = sample_user_model
    assert isinstance(model, Model)
    context = []
    session = model
    message = "Three word prompt"
    context.append({"role": "user", "content": message})
    session = session.chat(message)
    assert type(session) is Response
    assert isinstance(session.reply, str)
    assert session.reply == "I reject your 3 word prompt, and replace it with my own."


def test_list_models():
    inject()
    service = connect()
    assert isinstance(service, Service)
    assert service.list() == ["A", "B", "C"]


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

reply_to_hello_no_prompt = """
{"host": null, "model": "test-model", "messages": [{"role": "user", "content":
"Hello"}], "options": {}, "format": "", "extra": null}
""".lstrip()


def test_echo(sample_model, capfd):
    capfd.readouterr()

    resp = sample_model.chat("Hello")
    assert capfd.readouterr().out == ""

    for line in reply_to_hello.splitlines():
        assert len(line) <= 78

    resp = (sample_model | echo()).chat("Hello")
    assert remove_spinner(capfd.readouterr().out) == reply_to_hello

    resp = (sample_model | echo(spinner=False)).chat("Hello")
    assert capfd.readouterr().out == reply_to_hello

    for line in reply_to_hello_48.splitlines():
        assert len(line) <= 48

    resp = (sample_model | echo(width=48)).chat("Hello")
    assert remove_spinner(capfd.readouterr().out) == reply_to_hello_48

    for line in reply_to_hello_8.splitlines():
        assert len(line) <= 8 or " " not in line

    resp = (sample_model | echo(width=8)).chat("Hello")
    assert remove_spinner(capfd.readouterr().out) == reply_to_hello_8

    resp = (sample_model | echo(prompt=False)).chat("Hello")
    assert remove_spinner(capfd.readouterr().out) == reply_to_hello_no_prompt


def test_stats(sample_model, capfd):
    capfd.readouterr()

    (sample_model | stats()).chat("Hello")
    txt = remove_spinner(capfd.readouterr().out)
    pattern = r"prompt\s*:\s*\d+b,\s*reply\s*:\s*\d+t,\s*first\s*token\s*:\s*\d+(\.\d+)?s,\s*tokens\/s\s*:\s*\d+"
    assert re.search(pattern, txt), f"found: {txt}"

    capfd.readouterr()
    try:
        (sample_model | stats()).chat("FAIL(1)")
    except Exception:
        ...
    txt = remove_spinner(capfd.readouterr().out)
    assert txt == "- prompt : 7b, LLMError Exception raised\n"


def test_options(sample_model):
    reply = (sample_model | options(seed=12345)).chat("")
    context = [{"role": "user", "content": ""}]
    assert reply.reply == llm(None, test_model_name, context, {"seed": 12345}, "")


def test_system(sample_model):
    reply = sample_model.system("woof!").chat("")
    context = [{"role": "system", "content": "woof!"}, {"role": "user", "content": ""}]
    assert reply.reply == llm(None, test_model_name, context, {}, "")


class B(BaseModel):
    payload: int


class A(BaseModel):
    payload: B


def test_json_and_format(sample_model):
    reply = sample_model.chat("")
    context = [{"role": "user", "content": ""}]
    llm_reply = llm(None, test_model_name, context, {}, "")
    assert reply.reply == llm_reply
    assert reply.value is None

    reply = sample_model.chat("", middleware=format())
    context = [{"role": "user", "content": ""}]
    llm_reply = llm(None, test_model_name, context, {}, "json")
    assert reply.reply == llm_reply
    assert reply.value == json.loads(llm_reply)

    reply = sample_model.chat("", middleware=format(LLM))
    context = [{"role": "user", "content": ""}]
    llm_reply = llm(None, test_model_name, context, {}, LLM.model_json_schema())
    assert reply.reply == llm_reply
    assert reply.value == LLM(
        host=None,
        model=test_model_name,
        messages=context,
        options={},
        format=LLM.model_json_schema(),
        extra=None,
    )

    reply = sample_model.chat("", middleware=format({"type": "boolean"}))
    assert isinstance(reply.value, bool), f"expected bool, found {type(reply.value)}"

    for ty in [bool, int, float, str, list, dict]:
        reply = sample_model.chat("", middleware=format(ty))
        assert isinstance(reply.value, ty), f"expected {ty}, found {type(reply.value)}"

    reply = sample_model.chat("", middleware=format(tuple[int, bool]))
    assert reply.value == [99, False]

    reply = sample_model.chat("", middleware=format(list[int]))
    assert reply.value == [1, 2, 3]

    reply = sample_model.chat("", middleware=format(int | None))
    assert reply.value == 101

    reply = sample_model.chat("", middleware=format(A))
    assert reply.value == A(payload=B(payload=101))


def test_cache(sample_model, tmp_path):
    temp_file = tmp_path / "cache.db"
    mode = "a+"
    model = sample_model | cache(temp_file, mode)

    def replace(object, **kwargs):
        """compatible way to replace fields in a pydantic object"""
        return object.model_copy(update=kwargs)

    hello = "### Hello"
    assert len(model.children(hello)) == 0
    assert len(model.children()) == 0

    reply1 = model.chat(hello)
    assert len(model.children(hello)) == 1
    assert len(model.children()) == 1
    assert model.children(hello)[0] == replace(reply1, metrics=None)

    reply2 = model.chat(hello)
    assert len(model.children(hello)) == 2
    assert len(model.children()) == 2
    assert model.children(hello)[1] == replace(reply2, metrics=None)

    world = "### World"
    reply3 = reply2.chat(world)
    assert len(reply2.children()) == 1
    assert len(reply2.children(world)) == 1
    assert reply2.children(world)[0] == replace(reply3, metrics=None)

    # reset the cursor, to simulate a new execute
    sys.modules["haverscript.cache"].Cache.connections = {}
    model = sample_model | cache(temp_file, mode)

    assert len(model.children()) == 2
    reply1b = model.chat(hello)
    assert replace(reply1, metrics=None) == replace(reply1b, metrics=None)

    reply2b = model.chat(hello)

    assert replace(reply2, metrics=None) == replace(reply2b, metrics=None)
    # Check there was a difference to observe
    assert replace(reply1b, metrics=None) != replace(reply2b, metrics=None)

    # check the fresh flag gives a new value
    assert len(model.children()) == 2
    reply2c = model.chat(hello, middleware=fresh())
    assert replace(reply2, metrics=None) != replace(reply2c, metrics=None)
    assert len(model.children()) == 3

    # now test the read mode
    sys.modules["haverscript.cache"].Cache.connections = {}
    mode = "r"
    model = sample_model | cache(temp_file, mode)

    assert len(model.children()) == 3
    reply1b = model.chat(hello)

    scrub = dict(settings=None, parent=None, contexture=None)

    assert replace(reply1, metrics=None, **scrub) == replace(
        reply1b, metrics=None, **scrub
    )

    reply2b = model.chat(hello)

    assert replace(reply2, metrics=None, **scrub) == replace(
        reply2b, metrics=None, **scrub
    )

    # Check they are the same in read mode
    assert replace(reply1b, metrics=None) == replace(reply2b, metrics=None)

    sys.modules["haverscript.cache"].Cache.connections = {}
    mode = "a"
    model = sample_model | cache(temp_file, mode)

    hello = "### Hello"
    assert len(model.children(hello)) == 0
    assert len(model.children()) == 0

    reply1 = model.chat(hello)
    # Nothing to read, this is append mode
    assert len(model.children(hello)) == 0
    assert len(model.children()) == 0


def test_check(sample_model):
    # simple check
    assert repr(
        (sample_model | validate(lambda reply: "Squirrel" in reply)).chat("Squirrel")
    ).startswith("Response")

    # failing check
    with pytest.raises(LLMResultError):
        (
            sample_model
            | validate(lambda reply: "Squirrel" not in reply)
            | retry(stop=stop_after_attempt(5))
        ).chat("Squirrel")

    # chaining checks
    (
        sample_model
        | validate(lambda reply: "Squirrel" in reply)
        | validate(lambda reply: "Haggis" in reply)  # add on check for Haggis
        | validate(valid_json)
    ).chat(  # check output is valid JSON (the test stub used JSON for output)
        "Squirrel Haggis"
    )


def test_chat_middleware(sample_model: Model, capfd):
    with pytest.raises(LLMResultError):
        sample_model.chat(  # check output is valid JSON (the test stub used JSON for output)
            "Haggis", middleware=validate(lambda reply: "Squirrel" in reply)
        )

    result = (sample_model | options(fst="Hello")).chat(
        "Hello", middleware=options(foo="json")
    )
    assert '"options": {"foo": "json", "fst": "Hello"}' in repr(result)

    result = (sample_model | options(fst="Hello")).chat(
        "Hello", middleware=options(fst="World")
    )
    assert '"options": {"fst": "Hello"}' in repr(result)

    # check that middleware cleanly removes threads
    threads_before = len(threading.enumerate())
    capfd.readouterr()
    with pytest.raises(LLMResultError):
        result = sample_model.chat(  # check output is valid JSON (the test stub used JSON for output)
            "Haggis", middleware=validate(lambda reply: "Squirrel" in reply) | echo()
        )

    threads_after = len(threading.enumerate())

    assert remove_spinner(capfd.readouterr().out) == "\n> Haggis\n\n"
    assert threads_before == threads_after, "remaining thread outstanding"

    with pytest.raises(LLMResultError):
        result = sample_model.chat(  # check output is valid JSON (the test stub used JSON for output)
            "Haggis", middleware=validate(lambda reply: "Squirrel" in reply) | stats()
        )

    threads_after = len(threading.enumerate())
    assert (
        remove_spinner(capfd.readouterr().out)
        == "- prompt : 6b, LLMError Exception raised\n"
    )
    assert threads_before == threads_after, "remaining thread outstanding"


def valid_json(txt):
    try:
        json.loads(str(txt))
        return True
    except json.JSONDecodeError as e:
        raise False


def test_image(sample_model):
    image_src = f"{Path(__file__).parent}/../examples/images/edinburgh.png"
    prompt = "Describe this image"
    resp = sample_model.chat(prompt, images=[image_src])
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


def test_retry(sample_model):
    with pytest.raises(LLMError):
        sample_model.chat("FAIL(0)")

    extra = sample_model.chat("###", middleware=format()).value["extra"]
    model = sample_model | retry(stop=stop_after_attempt(5))
    model.chat(f"FAIL({extra+4})")


def test_validate(sample_model: Model):
    with pytest.raises(LLMError):
        (sample_model | validate(lambda txt: "$$$" in txt)).chat("...")

    (sample_model | validate(lambda txt: "$$$" in txt)).chat("$$$")

    extra = sample_model.chat("###", middleware=format()).value["extra"]

    with pytest.raises(LLMError):
        (
            sample_model
            | validate(lambda txt: f'"extra": {extra + 10}' in txt)
            | (retry(stop=stop_after_attempt(5)))
        ).chat("###")

    extra = sample_model.chat("###", middleware=format()).value["extra"]

    (
        sample_model
        | validate(lambda txt: f'"extra": {extra + 3}' in txt)
        | (retry(stop=stop_after_attempt(5)))
    ).chat("###")

    extra = sample_model.chat("###", middleware=format()).value["extra"]

    with pytest.raises(LLMError):
        # wrong order; retry is bellow validate.
        (
            sample_model
            | retry(stop=stop_after_attempt(5))
            | validate(lambda txt: f'"extra": {extra + 3}' in txt)
        ).chat("###")


#


md0 = "System"

md1 = """
> Hello
World
"""

md2 = """
System
> Hello
World
> Sprite
"""


def test_load(sample_model):
    session = sample_model.load(md0)
    assert isinstance(session, Model) and not isinstance(session, Response)

    session = sample_model.load(md1)
    assert isinstance(session, Response)
    assert session.contexture.system is None
    assert session.prompt == "Hello"
    assert session.reply == "World"

    session = sample_model.load(md2)
    assert isinstance(session, Response)
    assert session.contexture.system == "System"
    assert session.prompt == "Sprite"
    assert session.reply == ""

    session = sample_model.load(md2, complete=True)
    assert isinstance(session, Response)
    assert session.contexture.system == "System"
    assert session.prompt == "Sprite"
    context = [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "World"},
        {"role": "user", "content": "Sprite"},
    ]
    assert session.reply == llm(None, test_model_name, context, {}, "")


class UpperCase(Middleware):

    def upper_tokens(self, responses):
        for token in responses:
            if isinstance(token, str):
                yield token.upper()

    def invoke(self, request: Request, next: LanguageModel):
        prompt = request.prompt.model_copy(
            update=dict(content=request.prompt.content + " World")
        )
        request = request.model_copy(update=dict(prompt=prompt))
        responses = next.ask(request=request)
        return Reply(str(responses).upper())


def test_middleware(sample_model: Model):
    reply0 = sample_model.chat("Hello" + " World")

    session = sample_model | UpperCase()
    reply1 = session.chat("Hello")
    reply2 = (session | retry(stop=stop_after_attempt(5))).chat("Hello")

    assert reply0.reply.upper() == reply1.reply
    assert reply0.reply.upper() == reply2.reply


def test_Reply():
    """Test that Reply responses in a thread-safe manner"""

    def gen(xs):
        for x in xs:
            time.sleep(0.01)
            yield f"{x} "

    def input(n):
        return list(range(10))

    lmr = Reply(gen(input(10)))

    def consume():
        result = []
        for x in lmr.tokens():
            result.append(int(x))
        assert result == input(10)

    def metrics():
        assert lmr.metrics() is None

    ts = []
    for n in range(10):
        if n == 4:
            fun = metrics
        else:
            fun = consume
        ts.append(threading.Thread(target=fun))

    for t in ts:
        t.start()

    for t in ts:
        t.join()

    closing = [False, False, False, False]

    def close(n):
        closing[n] = True

    m1 = Reply(gen(input(10)))
    m1.after(lambda: close(0))
    m2 = Reply(gen(input(5)))
    m2.after(lambda: close(1))
    m3 = Reply(gen(input(7)))
    m3.after(lambda: close(2))
    m4 = Reply(gen(input(7)))
    m4.after(lambda: close(3))
    # we ignore m3, and only poke at m4
    m12 = m1 + m2

    assert str(m12).strip() == " ".join([str(i) for i in input(10) + input(5)])

    for token in m4:
        break  # just read one token

    assert closing == [True, True, False, False]


def test_cache_class(tmp_path):
    temp_file = tmp_path / "cache.db"
    cache = Cache(temp_file, "a+")
    system = "..."
    context = (
        Exchange(
            prompt=Prompt(content="Hello"),
            reply=AssistantMessage(content="World"),
        ),
        Exchange(
            prompt=Prompt(content="Hello2"),
            reply=AssistantMessage(content="World2"),
        ),
    )
    prompt = "Hello!"
    images = ["foo.png"]
    reply = "Wombat"
    options = {"model": "modal"}
    cache.insert_interaction(system, context, prompt, images, reply, options)
    cache.insert_interaction(system, context, prompt, images, reply, options)
    cache.insert_interaction(
        system, context, prompt + "..2", images, reply + "..2", options
    )

    # pretend that the database is read fresh
    cache.conn.execute("DELETE FROM blacklist;")

    assert cache.lookup_interactions(
        system, context, prompt, images, options, limit=None, blacklist=False
    ) == {INTERACTION(1): (prompt, images, reply)}

    cache.blacklist(INTERACTION(1))
    # Should not appear in our lookup (because of the blacklist)

    cache.insert_interaction(
        system, context, prompt, images, reply + " (Again)", options
    )

    assert (
        cache.lookup_interactions(
            system, context, prompt, images, options, limit=None, blacklist=True
        )
        == {}
    )
    assert cache.lookup_interactions(
        system, context, None, images, options, limit=None, blacklist=False
    ) == {
        INTERACTION(1): (prompt, images, reply),
        INTERACTION(2): (prompt + "..2", images, reply + "..2"),
        INTERACTION(3): (prompt, images, reply + " (Again)"),
    }

    assert cache.lookup_interactions(
        system, context, None, images, options, limit=1, blacklist=False
    ) == {
        INTERACTION(1): (prompt, images, reply),
    }
    result = subprocess.run(
        f'echo ".dump" | sqlite3 {temp_file}',
        shell=True,
        text=True,
        capture_output=True,
    )
    result = "\n".join([line.rstrip() for line in result.stdout.splitlines()])

    assert result == sql_dump


sql_dump = """
PRAGMA foreign_keys=OFF;
BEGIN TRANSACTION;
CREATE TABLE string_pool (
    id INTEGER PRIMARY KEY,
    string TEXT NOT NULL UNIQUE
);
INSERT INTO string_pool VALUES(1,'Hello');
INSERT INTO string_pool VALUES(2,'[]');
INSERT INTO string_pool VALUES(3,'World');
INSERT INTO string_pool VALUES(4,'Hello2');
INSERT INTO string_pool VALUES(5,'World2');
INSERT INTO string_pool VALUES(6,'Hello!');
INSERT INTO string_pool VALUES(7,'["foo.png"]');
INSERT INTO string_pool VALUES(8,'Wombat');
INSERT INTO string_pool VALUES(9,'...');
INSERT INTO string_pool VALUES(10,'{"model": "modal"}');
INSERT INTO string_pool VALUES(11,'Hello!..2');
INSERT INTO string_pool VALUES(12,'Wombat..2');
INSERT INTO string_pool VALUES(13,'Wombat (Again)');
CREATE TABLE context (
    id INTEGER PRIMARY KEY,
    prompt INTEGER  NOT NULL,       -- what was said to the LLM
    images INTEGER NOT NULL,        -- string of list of images
    reply INTEGER NOT NULL,         -- reply from the LLM
    context INTEGER,
    FOREIGN KEY (prompt)        REFERENCES string_pool(id),
    FOREIGN KEY (images)        REFERENCES string_pool(id),
    FOREIGN KEY (reply)         REFERENCES string_pool(id),
    FOREIGN KEY (context)       REFERENCES interactions(id)
);
INSERT INTO context VALUES(1,1,2,3,NULL);
INSERT INTO context VALUES(2,4,2,5,1);
INSERT INTO context VALUES(3,6,7,8,2);
INSERT INTO context VALUES(4,11,7,12,2);
INSERT INTO context VALUES(5,6,7,13,2);
CREATE TABLE interactions (
    id INTEGER PRIMARY KEY,
    system INTEGER,
    context INTEGER,
    parameters INTEGER NOT NULL,
    FOREIGN KEY (system)        REFERENCES string_pool(id),
    FOREIGN KEY (context)       REFERENCES context(id),
    FOREIGN KEY (parameters)    REFERENCES string_pool(id)
);
INSERT INTO interactions VALUES(1,9,3,10);
INSERT INTO interactions VALUES(2,9,4,10);
INSERT INTO interactions VALUES(3,9,5,10);
CREATE INDEX string_index ON string_pool(string);
CREATE INDEX context_prompt_index ON context(prompt);
CREATE INDEX context_images_index ON context(images);
CREATE INDEX context_reply_index ON context(reply);
CREATE INDEX context_context_index ON context(context);
CREATE INDEX interactions_system_index ON interactions(system);
CREATE INDEX interactions_context_index ON interactions(context);
CREATE INDEX interactions_parameters_index ON interactions(parameters);
COMMIT;
""".strip()


def test_transcript(sample_model: Model, tmp_path: str):
    temp_dir = tmp_path / "transcripts"

    transcripts = []

    model = sample_model | (transcript(temp_dir))
    session = model.chat("Hello")
    transcripts.append(session.render())
    session = session.chat("World")
    transcripts.append(session.render())

    model = model.system("SYSTEM")
    session = model.chat("Hello")
    transcripts.append(session.render())
    session = session.chat("World")
    transcripts.append(session.render())

    files = os.listdir(temp_dir)
    files = [f for f in files if os.path.isfile(os.path.join(temp_dir, f))]
    files.sort()

    # we also have the "latest.md" symbolic link
    assert len(files) - 1 == len(transcripts)

    for file, transcript_content in zip(files, transcripts):
        with open(os.path.join(temp_dir, file), "r", encoding="utf-8") as f:
            content = f.read()
            assert content == transcript_content


def test_markdown():
    prompt = Markdown()

    prompt += bullets(["Hello"])
    prompt += "Hello World"

    prompt += header("Feedback from Reader")

    prompt += text("Hello World")

    prompt += header("Feedback from Editor")
    prompt += bullets(["This is a test", "This is another test"])

    prompt += "Hello World"
    prompt += rule()
    prompt += "Hello World"
    prompt += bullets(["World 1", "World 2"], ordered=True)
    prompt += table(
        {"name": "Name", "age": "Age"},
        [
            {"name": "John", "age": 20},
            {"name": "Paul", "age": 22},
            {"name": "George", "age": 23},
            {"name": "Ringo", "age": 24},
        ],
    )

    prompt += code("print('Hello World')", language="python")

    prompt += quoted("Hello World with more text.\nanother line of text.")

    prompt += reply_in_json(LLM)

    assert (
        str(prompt)
        == """
- Hello

Hello World

# Feedback from Reader

Hello World

# Feedback from Editor

- This is a test
- This is another test

Hello World

---

Hello World

1. World 1
2. World 2

| Name   | Age |
|--------|-----|
| John   |  20 |
| Paul   |  22 |
| George |  23 |
| Ringo  |  24 |

```python
print('Hello World')
```

\"\"\"
Hello World with more text.
another line of text.
\"\"\"

Reply in JSON, using the following keys:

- "host" (str | None): The host of the LLM
- "model" (str | None): The model name
- "messages" (list | None): The conversation
- "options" (dict | None): The options
- "format" (str | dict): The format
- "extra" (str | None): Extra information

""".strip()
    )


def foo(i: int, s: str) -> bool:
    """foo returns True if i is less than or equal to 0

    args:
        i (int): The number to check
        s (str): A string to check

    returns:
        bool: True if x is less than or equal to 0
    """
    if i > 0:
        return False
    return True


def test_tool_schema():
    assert tool_schema(foo) == {
        "function": {
            "description": "foo returns True if i is less than or equal to 0",
            "name": "foo",
            "parameters": {
                "properties": {
                    "i": {
                        "type": "integer",
                    },
                    "s": {
                        "type": "string",
                    },
                },
                "required": [
                    "i",
                    "s",
                ],
                "type": "object",
            },
        },
        "type": "function",
    }


def test_toolcall(sample_model):
    response = sample_model.chat("ask tool", tools=tool(foo))
    assert response.contexture.context == (
        Exchange(
            prompt=Prompt(role="user", content="ask tool", images=()),
            reply=AssistantMessage(role="assistant", content="..."),
        ),
        Exchange(
            prompt=ToolResult(
                role="tool", results=(ToolReply(id="", name="foo", content="False"),)
            ),
            reply=AssistantMessage(
                role="assistant",
                content='{"host": null, "model": "test-model", "messages": [{"role": "user", "content": "ask tool"}, {"role": "assistant", "content": "..."}, {"role": "tool", "content": "False"}], "options": {}, "format": "", "extra": null}',
            ),
        ),
    )

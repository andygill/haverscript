import copy
import json
import logging
import re
import sqlite3
import textwrap
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, fields
from itertools import tee
from typing import Callable, Optional, Self, Tuple

import ollama

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Echo:
    width: int = 78

    def prompt(self, prompt: str):
        print()
        print("\n".join([f"> {line}" for line in prompt.splitlines()]))
        print()

    def reply(self, tokens, fresh: bool):
        for word in self.wrap(tokens):
            print(word, end="", flush=True)
        print()  # finish with a newline

    def regenerating(self):
        print()
        print("[Regenerating response]")
        print()

    def wrap(self, stream):

        line_width = 0  # the size of the commited line so far
        spaces = 0
        newline = "\n"
        prequel = ""

        for s in stream:

            for t in re.split(r"(\n|\S+)", s):

                if t == "":
                    continue

                if t.isspace():
                    if line_width + spaces + len(prequel) > self.width:
                        yield newline  # injected newline
                        line_width = 0
                        spaces = 0

                    if spaces > 0 and line_width > 0:
                        yield " " * spaces
                        line_width += spaces

                    spaces = 0
                    line_width += spaces + len(prequel)
                    yield prequel
                    prequel = ""

                    if t == "\n":
                        line_width = 0
                        yield newline  # actual newline
                    else:
                        spaces += len(t)
                else:
                    prequel += t

        if prequel != "":
            if line_width + spaces + len(prequel) > self.width:
                yield newline
                line_width = 0
                spaces = 0

            if spaces > 0 and line_width > 0:
                yield " " * spaces

            yield prequel

        return


@dataclass(frozen=True)
class Settings:
    """Local settings."""

    echo: Optional[Echo] = None

    cache: str = None

    outdent: bool = True

    def copy(self, **update):
        return Settings(
            **{
                f.name: update[f.name] if f.name in update else getattr(self, f.name)
                for f in fields(self)
            }
        )


@dataclass(frozen=True)
class Configuration:
    """Full Context and other arguments for the LLM."""

    host: str = None
    model: str = None
    options: dict = field(default_factory=dict)
    json: bool = False
    system: Optional[str] = None
    context: Tuple[  # list (using a tuple) of prompt response pairs
        Tuple[str, str], ...
    ] = ()

    def invoke(self, prompt: str, settings: Settings) -> str:
        messages = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        for exchange in self.context:
            messages.append({"role": "user", "content": exchange[0]})
            messages.append({"role": "assistant", "content": exchange[1]})
        messages.append({"role": "user", "content": prompt})
        options = copy.deepcopy(self.options)

        logging.info(
            f"Calling ollama chat with model={self.model}, stream={settings.echo}, messages={messages}, options={options}, json={self.json}"
        )

        response = services.ollama(self.host).chat(
            self.model,
            stream=settings.echo is not None,
            messages=messages,
            options=options,
            format="json" if self.json else "",
        )

        try:
            if settings.echo:
                tokens, results = tee(chunk["message"]["content"] for chunk in response)
                settings.echo.reply(tokens, fresh=True)
                text = "".join(results)
            else:
                text = response["message"]["content"]
            return text
        except Exception as e:
            # Slighty better messages. Should really have a type of reply for failure.
            if "ConnectError" in str(type(e)):
                print("Connection error (Check if ollama is running)")
            if "not found, try pulling it first":
                print(f"model not found (check ollama has {self.model} installed)")
            raise e

    def copy(self, **update):
        return Configuration(
            **{
                f.name: update[f.name] if f.name in update else getattr(self, f.name)
                for f in fields(self)
            }
        )

    def add_context(self, prompt: str, response: str):
        return self.copy(context=self.context + ((prompt, response),))

    def add_options(self, **options):
        return self.copy(
            options={
                key: value
                for key, value in {**self.options, **options}.items()
                if value is not None
            }
        )


@dataclass(frozen=True)
class Model(ABC):

    configuration: Configuration
    settings: Settings

    def chat(
        self,
        prompt: str,
    ) -> "Response":
        """
        Take a simple single prompt, and call a pre-specifed LLM,
        with optional context, returning a generated response.
        """
        if self.settings.outdent:
            prompt = textwrap.dedent(prompt).strip()

        logger.info(f"chat prompt {prompt}")
        if self.settings.echo:
            self.settings.echo.prompt(prompt)

        children = self.children(prompt)
        if len(children) > 0:
            response = children[-1]
            if self.settings.echo:
                self.settings.echo.reply(
                    re.split(r"(\S+)", response.reply), fresh=False
                )
            return response

        return self.invoke(prompt)

    def invoke(self, prompt: str) -> "Response":
        reply = self.configuration.invoke(prompt, settings=self.settings)
        if self.settings.cache is not None:
            services.cache(self.settings.cache).insert(
                self.configuration, prompt, reply
            )
        return self.response(prompt, reply, fresh=True)

    def response(self, prompt: str, reply: str, fresh: bool):
        return Response(
            configuration=self.configuration.add_context(prompt, reply),
            settings=self.settings,
            parent=self,
            fresh=fresh,
            _predicates=[],
        )

    def children(self, prompt: str = None):
        """Return all already cached replies to this prompt."""
        if self.settings.cache is None:
            return []
        cache = services.cache(self.settings.cache)
        replies = cache.lookup(self.configuration, prompt)
        return [
            self.response(prompt_, prose, fresh=False) for prompt_, prose in replies
        ]

    def copy(self, **update):
        return Model(
            **{
                f.name: update[f.name] if f.name in update else getattr(self, f.name)
                for f in fields(self)
            }
        )

    def echo(self, echo: Optional[bool | Echo] = True, width: int = 78) -> Self:
        """echo prompts and responses to stdout."""
        if echo == True:
            echo = Echo(width=width)
        elif echo == False:
            echo = None

        assert echo is None or isinstance(
            echo, Echo
        ), "echo() take a bool, or an Echo class"

        return self.copy(settings=self.settings.copy(echo=echo))

    def outdent(self, outdent: bool = True) -> Self:
        return self.copy(settings=self.settings.copy(outdent=outdent))

    def cache(self, filename: Optional[str] = None):
        """Set the cache filename for this model."""
        return self.copy(settings=self.settings.copy(cache=filename))

    def system(self, prompt: str) -> Self:
        """provide a system prompt."""
        return self.copy(configuration=self.configuration.copy(system=prompt))

    def json(self, json: bool = True):
        """request a json result."""
        return self.copy(configuration=self.configuration.copy(json=json))

    def options(
        self,
        **kwargs,
    ):
        """Options to pass to ollama, such as temperature and seed.

        num_ctx: int
        num_keep: int
        seed: int
        num_predict: int
        top_k: int
        top_p: float
        tfs_z: float
        typical_p: float
        repeat_last_n: int
        temperature: float
        repeat_penalty: float
        presence_penalty: float
        frequency_penalty: float
        mirostat: int
        mirostat_tau: float
        mirostat_eta: float
        penalize_newline: bool
        stop: Sequence[str]
        """
        return self.copy(configuration=self.configuration.add_options(**kwargs))


@dataclass(frozen=True)
class Response(Model):

    parent: Model
    fresh: bool  # was freshly generated (vs extracted from cache)
    _predicates: list[Callable[[Self], bool]]

    @property
    def prompt(self) -> str:
        assert len(self.configuration.context) > 0
        return self.configuration.context[-1][0]

    @property
    def reply(self) -> str:
        assert len(self.configuration.context) > 0
        return self.configuration.context[-1][1]

    def copy(self, **update):
        return Response(
            **{
                f.name: update[f.name] if f.name in update else getattr(self, f.name)
                for f in fields(self)
            }
        )

    def __str__(self):
        return self.reply

    def check(self, predicate, limit: Optional[int] = 10) -> Self:
        """A given predicate is applied to the Response.

        If the predicate is false, then a new Response is generated,
        using a new call to the LLM.

        There is an optional limit, which defaults to 10.
        """
        predicates = self._predicates + [predicate]
        session = self

        while not all([predicate(session) for predicate in predicates]):
            if limit is not None:
                if limit == 0:
                    raise RuntimeError(
                        "exceeded the count limit for redoing generation with predicate(s)"
                    )
                limit -= 1
            if self.settings.echo:
                self.settings.echo.regenerating()
            session = session.redo()

        return session.copy(_predicates=predicates)

    def redo(self):
        return self.parent.invoke(self.prompt)


def fresh(response):
    """Check a response is freshly generated (not obtained fromcache)."""
    return response.fresh


def valid_json(response):
    """Check to see if the response reply is valid JSON."""
    try:
        json.loads(response.reply)
        return True
    except json.JSONDecodeError:
        return False


def accept(response):
    """Ask the user if a response was acceptable."""
    answer = input("Accept? (Y/n)")
    return answer != "n"


class Cache:
    def __init__(self, filename: str) -> None:
        self.version = 1
        self.filename = filename

        self.conn = sqlite3.connect(
            filename, check_same_thread=sqlite3.threadsafety != 3
        )

        self.conn.executescript(
            f"""
            BEGIN;

            PRAGMA user_version = {self.version};

            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY,
                configuration TEXT NOT NULL, -- full configuration of the LLM
                prompt TEXT  NOT NULL,       -- what was said to the LLM
                reply TEXT NOT NULL         -- reply from the LLM
            );

            CREATE INDEX IF NOT EXISTS interactions_index ON interactions(configuration, prompt);

            COMMIT;
            """
        )

    def to_json(self, config):
        """We represent our saved config using a JSON string."""
        return json.dumps(asdict(config))

    def check_version(self):
        version = self.conn.execute("PRAGMA user_version").fetchone()[0]
        assert (
            version == self.version
        ), f"{repr(self.filename)} has schema version {version}, expecting version {self.version}."

    def insert(self, configuration: Configuration, prompt: str, reply: str) -> None:
        self.check_version()

        # storing a JSON string for the configuration is a work-in-process
        self.conn.execute(
            "INSERT INTO interactions (configuration, prompt, reply) VALUES (?, ?, ?)",
            (self.to_json(configuration), prompt, reply),
        )
        self.conn.commit()

    def lookup(
        self, configuration: Configuration, prompt: str = None
    ) -> list[(str, str)]:
        """Query to get the responses with this configuration, with optional prompt"""
        if prompt is None:
            return [
                (row[0], row[1])
                for row in self.conn.execute(
                    "SELECT prompt, reply FROM interactions WHERE configuration=? ORDER BY id",
                    (self.to_json(configuration),),
                ).fetchall()
            ]
        else:
            return [
                (prompt, row[0])
                for row in self.conn.execute(
                    "SELECT reply FROM interactions WHERE configuration=? AND prompt=? ORDER BY id",
                    (self.to_json(configuration), prompt),
                ).fetchall()
            ]


class Services:
    """Internal class with lazy instantiation of services"""

    def __init__(self) -> None:
        self._ollama = {}
        self._cache = {}

    def ollama(self, host=None):
        if host not in self._ollama:
            self._ollama[host] = ollama.Client(host)
        return self._ollama[host]

    def cache(self, filename):
        if filename not in self._cache:
            self._cache[filename] = Cache(filename)
        return self._cache[filename]


services = Services()


def connect(modelname: str, hostname: str = None) -> Model:
    """return a model that uses the given model name."""
    return Model(
        configuration=Configuration(model=modelname, host=hostname), settings=Settings()
    )

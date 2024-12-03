import copy
import json
import logging
import re
import sqlite3
import textwrap
import threading
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field, fields
from itertools import tee
from types import GeneratorType
from typing import AnyStr, Callable, Optional, Self, Tuple, NoReturn, Generator
from tenacity import Retrying, RetryError

import ollama
from yaspin import yaspin
from frozendict import frozendict

from .exceptions import *

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Echo:
    width: int = 78
    echo_prompt: bool = True

    def prompt(self, prompt: str):
        if self.echo_prompt:
            print()
            print("\n".join([f"> {line}" for line in prompt.splitlines()]))
            print()

    def reply(self, tokens, fresh: bool):

        tokens = self.wrap(tokens)

        first_token_available = threading.Event()
        first_token = [None]  # Use list to allow modification in nested scope

        def get_first_token():
            try:
                first_token[0] = next(tokens)
            except Exception as e:
                first_token[0] = e
            finally:
                first_token_available.set()

        threading.Thread(target=get_first_token).start()

        with yaspin() as spinner:
            first_token_available.wait()

        if first_token[0] is not None:
            if isinstance(first_token[0], Exception):
                raise first_token[0]
            print(first_token[0], end="", flush=True)

        for token in tokens:
            print(token, end="", flush=True)
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

    retry: dict | None = None

    service: "ServiceProvider" = None

    def copy(self, **update):
        return Settings(
            **{
                f.name: update[f.name] if f.name in update else getattr(self, f.name)
                for f in fields(self)
            }
        )


@dataclass(frozen=True)
class Configuration:
    """Full Context and other arguments for the LLM. Needs to be serializable."""

    model: str = None
    options: frozendict = field(default_factory=frozendict)
    json: bool = False
    system: Optional[str] = None
    context: Tuple[  # list (using a tuple) of prompt+images response triples
        Tuple[str, Tuple[str, ...], str], ...
    ] = ()
    images: Tuple[str, ...] = ()  # tuple of images

    def copy(self, **update):
        return Configuration(
            **{
                f.name: update[f.name] if f.name in update else getattr(self, f.name)
                for f in fields(self)
            }
        )

    def add_context(self, prompt: str, response: str):
        return self.copy(
            context=self.context + ((prompt, self.images, response),), images=()
        )

    def add_options(self, **options):
        return self.copy(
            options=frozendict(
                {
                    key: value
                    for key, value in {**self.options, **options}.items()
                    if value is not None
                }
            )
        )

    def add_image(self, image):
        return self.copy(images=self.images + (image,))


def _canonical_string(string, postfix="\n"):
    """Adds a newline to a string if needed, for outputting to a file."""
    if not string:
        return string
    if not string.endswith(postfix):
        overlap_len = next(
            (i for i in range(1, len(postfix)) if string.endswith(postfix[:i])), 0
        )
        string += postfix[overlap_len:]
    return string


@dataclass(frozen=True)
class Metrics(ABC):
    total_duration: int  # time spent generating the response
    load_duration: int  # time spent in nanoseconds loading the model
    prompt_eval_count: int  # number of tokens in the prompt
    prompt_eval_duration: int  # time spent in nanoseconds evaluating the prompt
    eval_count: int  # number of tokens in the response
    eval_duration: int  # time in nanoseconds spent generating the response


@dataclass(frozen=True)
class Metrics(ABC):
    pass


@dataclass(frozen=True)
class Service(ABC):
    service: "ServiceProvider"

    def list(self):
        return self.service.list()

    def model(self, model) -> "Model":
        return Model(
            configuration=Configuration(model=model),
            settings=Settings(service=self.service),
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

        if self.settings.cache is not None:
            cache = services.cache(self.settings.cache)
            prose = cache.next(self, prompt)
            if prose:
                if self.settings.echo:
                    self.settings.echo.reply(re.split(r"(\S+)", prose), fresh=False)
                return self.response(prompt, prose, fresh=False)

        return self.invoke(prompt)

    def invoke(self, prompt: str) -> "Response":

        settings = self.settings

        if settings.retry is not None:
            try:
                for attempt in Retrying(**settings.retry):
                    with attempt:
                        return self._invoke(prompt)

            except RetryError as e:
                raise LLMError()

        else:
            return self._invoke(prompt)

    def _invoke(self, prompt: str) -> "Response":

        settings = self.settings

        response = self.settings.service.chat(
            configuration=self.configuration,
            prompt=prompt,
            stream=settings.echo is not None,
        )

        assert isinstance(
            response, GeneratorType
        ), f"response : {type(response)}, expecting GeneratorType"

        response, metadata = tee(response)
        response = (token for token in response if isinstance(token, str))
        metadata = (token for token in metadata if isinstance(token, Metrics))
        if settings.echo:
            tokens, response = tee(response)
            settings.echo.reply(tokens, fresh=True)

        reply = "".join([r for r in response])

        metrics = None
        for meta in metadata:
            if isinstance(meta, Metrics):
                metrics = meta

        response = self.response(prompt, reply, fresh=True, metrics=metrics)

        if self.settings.cache is not None:
            services.cache(self.settings.cache).insert(response)

        return response

    def response(
        self,
        prompt: str,
        reply: str,
        fresh: bool,
        metrics: Metrics | None = None,
    ):
        assert isinstance(prompt, str)
        assert isinstance(reply, str)
        assert isinstance(fresh, bool)
        assert isinstance(metrics, (Metrics, type(None)))
        return Response(
            configuration=self.configuration.add_context(prompt, reply),
            settings=self.settings,
            parent=self,
            fresh=fresh,
            metrics=metrics,
            _predicates=(),
        )

    def children(self, prompt: str = None):
        """Return all already cached replies to this prompt."""
        if self.settings.cache is None:
            return []
        cache = services.cache(self.settings.cache)
        replies = cache.lookup(self, prompt)
        return [
            self.response(prompt_, prose, fresh=False) for prompt_, prose in replies
        ]

    def render(self) -> str:
        """Return a markdown string of the context."""
        return _canonical_string(self.configuration.system or "")

    def copy(self, **update):
        return Model(
            **{
                f.name: update[f.name] if f.name in update else getattr(self, f.name)
                for f in fields(self)
            }
        )

    def echo(
        self,
        echo: Optional[bool | Echo] = True,
        width: int = 78,
        echo_prompt: bool = True,
    ) -> Self:
        """echo prompts and responses to stdout."""
        if echo == True:
            echo = Echo(width=width, echo_prompt=echo_prompt)
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

    def retry_policy(self, **options) -> Self:
        """retry uses tenacity to wrap the LLM request-response action in retry options."""
        return self.copy(settings=self.settings.copy(retry=options))

    def system(self, prompt: str) -> Self:
        """provide a system prompt."""
        return self.copy(configuration=self.configuration.copy(system=prompt))

    def json(self, json: bool = True):
        """request a json result."""
        return self.copy(configuration=self.configuration.copy(json=json))

    def load(self, markdown: str, complete: bool = False) -> Self:
        """Read markdown as system + prompt-reply pairs."""

        lines = markdown.split("\n")
        result = []
        if not lines:
            return self

        current_block = []
        current_is_quote = lines[0].startswith("> ")
        starts_with_quote = current_is_quote

        for line in lines:
            is_quote = line.startswith("> ")
            # Remove the '> ' prefix if it's a quote line
            line_content = line[2:] if is_quote else line
            if is_quote == current_is_quote:
                current_block.append(line_content)
            else:
                result.append("\n".join(current_block))
                current_block = [line_content]
                current_is_quote = is_quote

        # Append the last block
        if current_block:
            result.append("\n".join(current_block))

        model = self

        if not starts_with_quote:
            if sys_prompt := result[0].strip():
                # only use non-empty system prompts
                model = model.system(sys_prompt)
            result = result[1:]

        while result:
            prompt = result[0].strip()
            if len(result) == 1:
                reply = ""
            else:
                reply = result[1].strip()

            if complete and reply in ("", "..."):
                model = model.chat(prompt)
            else:
                model = model.response(prompt, reply, fresh=False)

            result = result[2:]

        return model

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

    def image(self, image: AnyStr):
        """image must be bytes, path-like object, or file-like object"""
        return self.copy(configuration=self.configuration.add_image(image))

    def middleware(self, f: Callable[["LanguageModel"], "Middleware"]):
        return self.copy(settings=self.settings.copy(service=f(self.settings.service)))


@dataclass(frozen=True)
class Response(Model):

    parent: Model
    fresh: bool  # was freshly generated (vs extracted from cache)
    metrics: Metrics | None

    _predicates: tuple  # [Callable[[Self], bool]]

    @property
    def prompt(self) -> str:
        assert len(self.configuration.context) > 0
        return self.configuration.context[-1][0]

    @property
    def reply(self) -> str:
        assert len(self.configuration.context) > 0
        return self.configuration.context[-1][2]

    @property
    def value(self) -> str:
        """return a value of the reply in its requested format"""
        if self.configuration.json:
            try:
                return json.loads(self.reply)
            except json.JSONDecodeError as e:
                return None

        return self.reply

    def render(self) -> str:
        """Return a markdown string of the context."""

        context = _canonical_string(self.parent.render(), postfix="\n\n")

        if self.prompt:
            prompt = (
                "".join([f"> {line}\n" for line in self.prompt.splitlines()]) + "\n"
            )
        else:
            prompt = ">\n\n"

        reply = self.reply or ""

        return context + prompt + _canonical_string(reply.strip())

    def json_value(self, limit: int | None = 10) -> dict:
        return self.check(valid_json, limit=limit).value

    def copy(self, **update):
        return Response(
            **{
                f.name: update[f.name] if f.name in update else getattr(self, f.name)
                for f in fields(self)
            }
        )

    def __str__(self):
        return self.reply

    def reject(self, message: str | None = None) -> NoReturn:
        if message:
            raise LLMResultError(message)
        raise LLMResultError()

    def check(self, predicate, limit: Optional[int] = 10) -> Self:
        """A given predicate is applied to the Response.

        If the predicate is false, then a new Response is generated,
        using a new call to the LLM.

        There is an optional limit, which defaults to 10.
        """
        session = self

        while not predicate(session):
            if limit is not None:
                if limit == 0:
                    self.reject(
                        "exceeded the count limit for redoing generation with predicate(s)"
                    )
                limit -= 1
            if self.settings.echo:
                self.settings.echo.regenerating()
            session = session.redo()

        return session.copy(_predicates=self._predicates + ((predicate, limit),))

    def redo(self) -> Self:
        """Rerun a chat with the previous prompt.

        If the Response has check(s), they are also re-run.
        If the Response used a seed, the seed is incremented.
        """
        model = self.parent
        if "seed" in model.configuration.options:
            # if we have set a seed, do not redo with the same seed,
            # because you would get the same result (even if cached).
            model = model.options(seed=model.configuration.options["seed"] + 1)
        previous = model.invoke(self.prompt)
        for pred, limit in self._predicates:
            previous = previous.check(pred, limit=limit)
        return previous


def fresh(response):
    """Check a response is freshly generated (not obtained fromcache)."""
    return response.fresh


def valid_json(response):
    """Check to see if the response reply is valid JSON."""
    return response.value is not None


def accept(response):
    """Ask the user if a response was acceptable."""
    answer = input("Accept? (Y/n)")
    return answer != "n"


class Cache:
    def __init__(self, filename: str) -> None:
        self.version = 2
        self.filename = filename
        self.local: dict[Response, int] = {}
        self.cursor: dict[tuple[Response, str], list[tuple[str, str]]] = {}

        self.conn = sqlite3.connect(
            filename, check_same_thread=sqlite3.threadsafety != 3
        )

        self.conn.executescript(
            f"""
            BEGIN;

            PRAGMA user_version = {self.version};

            CREATE TABLE IF NOT EXISTS string_pool (
                id INTEGER PRIMARY KEY,
                string TEXT NOT NULL UNIQUE
            );

            CREATE INDEX IF NOT EXISTS string_index ON string_pool(string);

            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY,
                configuration INTEGER NOT NULL, -- configuration without system and context
                system INTEGER,                 
                context INTEGER,                
                prompt INTEGER  NOT NULL,       -- what was said to the LLM
                reply INTEGER NOT NULL,         -- reply from the LLM
                FOREIGN KEY (configuration) REFERENCES string_pool(id),
                FOREIGN KEY (system)        REFERENCES string_pool(id),
                FOREIGN KEY (context)       REFERENCES interactions(id),
                FOREIGN KEY (prompt)        REFERENCES string_pool(id),
                FOREIGN KEY (reply)         REFERENCES string_pool(id)
            );

            CREATE INDEX IF NOT EXISTS interactions_configuration_index ON interactions(configuration);
            CREATE INDEX IF NOT EXISTS interactions_system_index ON interactions(system);
            CREATE INDEX IF NOT EXISTS interactions_context_index ON interactions(context);
            CREATE INDEX IF NOT EXISTS interactions_prompt_index ON interactions(prompt);

            CREATE VIEW IF NOT EXISTS interactions_with_strings AS
            SELECT 
                interactions.id,
                (SELECT string FROM string_pool WHERE id = interactions.configuration) AS configuration,
                COALESCE((SELECT string FROM string_pool WHERE id = interactions.system), NULL) AS system,
                interactions.context,
                (SELECT string FROM string_pool WHERE id = interactions.prompt) AS prompt,
                (SELECT string FROM string_pool WHERE id = interactions.reply) AS reply
            FROM 
                interactions
            ;
            

            COMMIT;
            """
        )

    def to_json(self, config):
        """We represent our saved config using a JSON string."""
        assert isinstance(config, Configuration)
        return json.dumps(
            {
                k: v
                for k, v in asdict(config).items()
                # system and context are handled seperately
                # images are not (yet) supported
                if k not in ["system", "context", "images"]
            }
        )

    def check_version(self):
        version = self.conn.execute("PRAGMA user_version").fetchone()[0]
        assert (
            version == self.version
        ), f"{repr(self.filename)} has schema version {version}, expecting version {self.version}."

    def _string_lookup(self, string: str, update: bool = False) -> int:
        if string is None:
            return None
        if update:
            self.conn.execute(
                "INSERT OR IGNORE INTO string_pool (string) VALUES (?)", (string,)
            )
            self.conn.commit()

        # Retrieve the id of the string (whether it was newly inserted or already exists)
        row = self.conn.execute(
            "SELECT id FROM string_pool WHERE string = ?", (string,)
        ).fetchone()
        if row:
            return row[0]  # Return the id of the string
        else:
            return None

    def _interactions_dict(
        self,
        configuration: Configuration,
        context: Model | None = None,
        prompt: str | None = None,
        reply: str | None = None,
        update: bool = False,
    ):
        args = {
            "configuration": self._string_lookup(
                self.to_json(configuration), update=update
            ),
            "system": self._string_lookup(configuration.system, update=update),
            "context": self._response_lookup(context, update=update),
        }

        if prompt:
            args["prompt"] = self._string_lookup(prompt, update=update)
        if reply:
            args["reply"] = self._string_lookup(reply, update=update)

        return args

    def _interactions_prompt(
        self,
        elements,
        args,
    ):
        nullable_compares = []
        compares = []
        for k, v in args.items():
            if v is None:
                nullable_compares.append(f"{k} IS NULL")
            else:
                compares.append(f"{k} == :{k}")

        prompt = (
            "SELECT "
            + ",".join(elements)
            + " FROM interactions WHERE  "
            + " AND ".join(compares + nullable_compares)
        )

        return prompt

    def _response_lookup(self, response: Model, update: bool = False) -> int:

        if isinstance(response, Response):
            if response in self.local:
                return self.local[response]

            args = self._interactions_dict(
                response.configuration,
                response.parent,
                response.prompt,
                response.reply,
                update=update,
            )

            found = self.conn.execute(
                self._interactions_prompt(["id"], args), args
            ).fetchone()

            if found is not None:
                ix = found[0]
                self.local[response] = ix
                return ix

            if update:
                ix = self.conn.execute(
                    """
                    INSERT INTO interactions ( configuration,  system,  context,  prompt,  reply)  
                                      VALUES (:configuration, :system, :context, :prompt, :reply)
                    """,
                    args,
                ).lastrowid
                self.local[response] = ix
                return ix

        return None

    def insert(self, response: Response) -> None:
        self.check_version()
        self._response_lookup(response, update=True)
        self.conn.commit()

    def lookup(self, model: Model, prompt: str = None) -> list[tuple[str, str]]:
        # looking for all the times this configuration/model was used with a prompt
        if isinstance(model, Response):
            args = self._interactions_dict(
                model.configuration, context=model, prompt=prompt
            )
        else:
            args = self._interactions_dict(
                model.configuration, context=None, prompt=prompt
            )

        return [
            (row[0], row[1])
            for row in self.conn.execute(
                self._interactions_prompt(
                    [
                        "(SELECT string FROM string_pool WHERE string_pool.id == prompt) as prompt",
                        "(SELECT string FROM string_pool WHERE string_pool.id == reply) as reply",
                    ],
                    args,
                ),
                args,
            ).fetchall()
        ]

    def next(self, model: Model, prompt: str) -> str | None:
        if (model, prompt) not in self.cursor:
            replies = self.lookup(model, prompt)
            self.cursor[model, prompt] = replies
        else:
            replies = self.cursor[model, prompt]

        if replies:
            return replies.pop(0)[1]

        return None


class LanguageModel(ABC):
    """Base class for anything that chats, that is takes a configuration and prompt and returns token(s)."""

    @abstractmethod
    def chat(
        self, configuration: Configuration, prompt: str, stream: bool
    ) -> Generator[str | Metrics, None, None]:
        """Call the chat method of an LLM.

        prompt is the main text
        configuration is (structured) context
        stream, if true, is a request for a streamed reply

        Returns a GeneratorType / Generator of str or Metrics.

        We use Generator explicitly because this is a consumable
        value, that is, iterating over it consumes it.
        """
        pass


class ServiceProvider(LanguageModel):
    """A ServiceProvider is a LanguageModel that serves specific models."""

    def list(self) -> list[str]:
        models = self.client[self.hostname].list()
        assert "models" in models
        return [model["name"] for model in models["models"]]


@dataclass
class Middleware(LanguageModel):
    """Middleware is a LanguageModel that has next down-the-pipeline LanguageModel."""

    next: LanguageModel


class Services:
    """Internal class with lazy instantiation of services"""

    def __init__(self) -> None:
        self._model_providers = {}
        self._cache = {}

    def cache(self, filename):
        if filename not in self._cache:
            self._cache[filename] = Cache(filename)
        return self._cache[filename]


services = Services()


@dataclass(frozen=True)
class OllamaMetrics(Metrics):
    total_duration: int  # time spent generating the response
    load_duration: int  # time spent in nanoseconds loading the model
    prompt_eval_count: int  # number of tokens in the prompt
    prompt_eval_duration: int  # time spent in nanoseconds evaluating the prompt
    eval_count: int  # number of tokens in the response
    eval_duration: int  # time in nanoseconds spent generating the response


class Ollama(ServiceProvider):
    client = {}

    def __init__(self, hostname: str | None = None) -> None:
        self.hostname = hostname
        if hostname not in Ollama.client:
            Ollama.client[hostname] = ollama.Client(host=hostname)

    def list(self) -> list[str]:
        models = self.client[self.hostname].list()
        assert "models" in models
        return [model["name"] for model in models["models"]]

    def _suggestions(self, e: Exception):
        # Slighty better message. Should really have a type of reply for failure.
        if "ConnectError" in str(type(e)):
            print("Connection error (Check if ollama is running)")
        return e

    def chat(self, configuration: Configuration, prompt: str, stream: bool):
        messages = []

        if configuration.system:
            messages.append({"role": "system", "content": configuration.system})

        for pmt, imgs, resp in configuration.context:
            messages.append(
                {"role": "user", "content": pmt}
                | ({"images": list(imgs)} if imgs else {})
            )
            messages.append({"role": "assistant", "content": resp})

        messages.append(
            {"role": "user", "content": prompt}
            | ({"images": list(configuration.images)} if configuration.images else {})
        )

        try:
            response = self.client[self.hostname].chat(
                configuration.model,
                stream=stream,
                messages=messages,
                options=copy.deepcopy(configuration.options),
                format="json" if configuration.json else "",
            )

            if isinstance(response, GeneratorType):
                try:
                    for chunk in response:
                        if chunk["done"]:
                            yield OllamaMetrics(
                                **{
                                    k: chunk[k]
                                    for k in OllamaMetrics.__dataclass_fields__.keys()
                                }
                            )
                        yield from chunk["message"]["content"]
                except Exception as e:
                    raise self._suggestions(e)
            else:
                assert isinstance(response["message"]["content"], str)
                yield response["message"]["content"]
                yield OllamaMetrics(
                    **{
                        k: response[k]
                        for k in OllamaMetrics.__dataclass_fields__.keys()
                    }
                )

        except Exception as e:
            raise self._suggestions(e)


def connect(
    modelname: str | None = None,
    hostname: str | None = None,
    service: ServiceProvider | None = None,
) -> Model | Service:
    """return a model that uses the given model name."""

    assert (
        hostname is None or service is None
    ), "can not use hostname with a provided service provider (pass hostname directly)"

    if service is None:
        service = Ollama(hostname)

    service = Service(service)

    if modelname is not None:
        return service.model(modelname)

    return service

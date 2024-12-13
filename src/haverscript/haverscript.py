import json
import logging
import sqlite3
import textwrap
from abc import ABC
from dataclasses import asdict, dataclass, field, fields
from itertools import tee
from typing import AnyStr, Callable, Optional, Self, Tuple, NoReturn

from yaspin import yaspin
from frozendict import frozendict

from .exceptions import *
from .languagemodel import *
from .middleware import *
from .ollama import Ollama
from .cache import *
from .render import render_system, render_interaction

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Settings:
    """Local settings."""

    cache: str = None

    outdent: bool = True

    service: "Middleware" = None

    middleware: Middleware = field(default_factory=EmptyMiddleware)

    def copy(self, **update):
        return Settings(
            **{
                f.name: update[f.name] if f.name in update else getattr(self, f.name)
                for f in fields(self)
            }
        )


@dataclass(frozen=True)
class Service(ABC):
    service: "ServiceProvider"

    def list(self):
        return self.service.list()

    def model(self, model) -> "Model":
        return Model(
            #            configuration=Configuration(),
            settings=Settings(service=self.service, middleware=ModelMiddleware(model)),
            contexture=LanguageModelContexture(model=model),
        )


@dataclass(frozen=True)
class Model(ABC):

    #    configuration: Configuration
    settings: Settings
    contexture: LanguageModelContexture

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

        if self.settings.cache is not None:
            cache = services.cache(self.settings.cache)
            prose = cache.next(self, prompt)
            if prose:
                return self.response(prompt, prose, fresh=False)

        return self.invoke(prompt)

    def _chat_args(self):
        return dict(
            options=self.contexture.options.copy(),
            json=self.contexture.format,
            system=self.contexture.system,
            context=self.contexture.context,
            images=self.contexture.images,
        )

    def request(self, prompt: str | None) -> LanguageModelRequest:
        return LanguageModelRequest(
            contexture=self.contexture, prompt=prompt, stream=False
        )

    def invoke(self, prompt: str | None) -> "Response":

        middleware: Middleware = self.settings.middleware

        response = middleware.invoke(
            request=self.request(prompt), next=self.settings.service
        )

        assert prompt is not None, "Can not build a response with no prompt"

        assert isinstance(
            response, LanguageModelResponse
        ), f"response : {type(response)}, expecting LanguageModelResponse"

        # Run for any continuations before returning
        response.close()

        response = self.response(
            prompt, str(response), fresh=True, metrics=response.metrics()
        )

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
            settings=self.settings,
            contexture=self.contexture.append_exchange(
                LanguageModelExchange(
                    prompt=prompt, images=self.contexture.images, reply=reply
                )
            ),
            parent=self,
            fresh=fresh,
            metrics=metrics,
            _predicates=(),
        )

    def children(self, prompt: str | None = None, images: list[str] | None = None):
        """Return all already cached replies to this prompt."""

        service = self.settings.service
        first = self.settings.middleware.first()

        if not isinstance(first, CacheMiddleware):
            # only top-level cache can be interrogated.
            raise LLMInternalError(
                ".children(...) method needs cache to be final middleware"
            )

        replies = first.children(self.request(prompt))

        return [
            self.response(prompt_, prose, fresh=False)
            for prompt_, imgs, prose in replies
        ]

    def render(self) -> str:
        """Return a markdown string of the context."""
        return render_system(self.contexture.system)

    def copy(self, **update):
        return Model(
            **{
                f.name: update[f.name] if f.name in update else getattr(self, f.name)
                for f in fields(self)
            }
        )

    # middleware methods

    def echo(self, width: int = 78, prompt: bool = True, spinner: bool = True) -> Self:
        """echo prompts and responses to stdout."""
        return self.middleware(echo(width, prompt, spinner))

    def cache(self, filename: str, mode: str | None = "a+") -> Self:
        """Set the cache filename for this model."""
        return self.middleware(cache(filename, mode))

    def retry(self, **options) -> Self:
        """Set the cache filename for this model."""
        return self.middleware(retry(**options))

    def stats(self):
        return self.middleware(stats())

    def transcript(self, dirname: str):
        return self.middleware(transcript(dirname))

    def validate(self, predicate: Callable[[str], bool]):
        return self.middleware(validate(predicate))

    # Content methods

    def outdent(self, outdent: bool = True) -> Self:
        return self.copy(settings=self.settings.copy(outdent=outdent))

    def system(self, system: str) -> Self:
        """provide a system prompt."""
        return self.copy(
            contexture=self.contexture.model_copy(update=dict(system=system))
        )

    def json(self, json: bool = True):
        """request a json result."""
        assert isinstance(json, bool)
        format = "json" if json else ""
        return self.copy(
            contexture=self.contexture.model_copy(update=dict(format=format))
        )

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
        return self.copy(contexture=self.contexture.add_options(**kwargs))

    def image(self, image: AnyStr):
        """image must be bytes, path-like object, or file-like object"""
        return self.copy(contexture=self.contexture.append_image(image))

    def middleware(self, after: Middleware):
        return self.copy(
            settings=self.settings.copy(middleware=self.settings.middleware | after)
        )


@dataclass(frozen=True)
class Response(Model):

    parent: Model
    fresh: bool  # was freshly generated (vs extracted from cache)
    metrics: Metrics | None

    _predicates: tuple  # [Callable[[Self], bool]]

    @property
    def prompt(self) -> str:
        assert len(self.contexture.context) > 0
        return self.contexture.context[-1].prompt

    @property
    def reply(self) -> str:
        assert len(self.contexture.context) > 0
        return self.contexture.context[-1].reply

    @property
    def value(self) -> str:
        """return a value of the reply in its requested format"""
        try:
            return json.loads(self.reply)
        except json.JSONDecodeError as e:
            return None

    def render(self) -> str:
        """Return a markdown string of the context."""
        return render_interaction(self.parent.render(), self.prompt, self.reply)

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

    def redo(self) -> Self:
        """Rerun a chat with the previous prompt.

        If the Response has check(s), they are also re-run.
        If the Response used a seed, the seed is incremented.
        """
        model = self.parent
        if "seed" in model.contexture.options:
            # if we have set a seed, do not redo with the same seed,
            # because you would get the same result (even if cached).
            model = model.options(seed=model.contexture.options["seed"] + 1)
        return model.invoke(self.prompt)


def valid_json(response):
    """Check to see if the response reply is valid JSON."""
    return response.value is not None


def accept(response):
    """Ask the user if a response was acceptable."""
    answer = input("Accept? (Y/n)")
    return answer != "n"


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



# Middleware

Middleware is a mechansim to have fine control over everything between
calling `.chat` and Haverscript calling the LLM.
As a example, consider the creation of a session.

```python
session = connect("mistral") | echo()
```

We use `|` to create a pipeline for actions to be taken.
The reply from the LLM flows left to right, and the prompt
to invoke the LLM flows from right to left. In this case,
there is only `echo`, but consider two examples using `cache`.

```python
#  LLM <---- PROMPT GOES RIGHT TO LEFT <----- chat
connect("mistral") | echo() | cache("cache.db") 
#  LLM ----> REPLY GOES LEFT TO RIGHT ------> Response
```
In this example, we check the cache for any existing answer, then call echo
only if we do not have a cache hit.

```python
#  LLM <---- PROMPT GOES RIGHT TO LEFT <----- chat
connect("mistral") | cache("cache.db") | echo() 
#  LLM ----> REPLY GOES LEFT TO RIGHT ------> Response
```

In this second example, we always echo, even if the
cache hits. 

The user has the capability to choose the middleware stack they want
for their application. 

There are two ways of defining middleware. First as part of the overall session
(as above), or as part of a chat. In this second case, the middleware is only append
for this chat, and is not part of any context. 

```python
session.chat("Hello", middlware=echo())
```

# Middleware in Haverscript

Haverscript provides following middleware

| Middleware | Purpose | Class |
|------------|---------|-------|
| model      | Request a specific model be used            | configuration | 
| options    | Set specific LLM options (such as seed)     | configuration |
| format     | Set specific format for output              | configuration |
| echo       | Print prompt and reply                      | observation |
| stats      | Print basic stats about LLM                 | observation |
| trace      | Log requests and responses                  | observation |
| transcript | Store a complete transcript of every call   | observation |
| retry      | retry on failure (using tenacity)           | reliablity |
| validation | Fail under given condition                  | reliablity |
| cache      | Store and/or query prompt-reply pairs in DB | efficency | 
| fresh      | Request a fresh reply (not cached)          | efficency |
| meta       | Support for generalized prompt and response transformations | generalization |

## Configuration Middleware

```python
def model(model_name: str) -> Middleware: 
def options(**kwargs) -> Middleware:
def format(schema: Type[BaseModel] | None = None) -> Middleware:
```
    
* `model` is automatically appended to the start of the middleware by the call to
`connect`. 
* `options` allows options, such as temperature, the seed, and top_k,
to be set. 
* `format` requests the the output be formatted in JSON, and
automatically parse the output. If no type is provided, then the result is a
JSON `dict`.  If a BaseModel (from pydantic) type is provided then the schema
of this specific BaseModel class is used, and the class is parsed. In either
case, `Response.value` is used to access the parsed result.

See [options](examples/options/README.md) for an example of using `format`.
See [formatting](examples/format/README.md) for an example of using `format`.

## Observation Middleware

There are four middleware adapters for observation.

```python
def echo(width: int = 78, prompt: bool = True, spinner: bool = True) -> Middleware:
def stats() -> Middleware:
def trace(level: int = log.DEBUG) -> Middleware:
def transcript(dirname: str) -> Middleware:
```

* `echo` turns of echo of prompt and reply. There is a spinner (⠧) which is
used when waiting for a response from the LLM.
* `stats` prints based stats (token counts, etc) to the screen
* `trace` uses pythons logging to log all prompts and responses.
* `transcript` stores all prompt-response pairs, including context, in a sub-directory.

## Reliablity Middleware

```python
def retry(**options) -> Middleware:
    """retry uses tenacity to wrap the LLM request-response action in retry options."""
def validate(predicate: Callable[[str], bool]) -> Middleware:
    """validate the response as middleware. Can raise as LLMResultError"""
```

## Efficency Middleware

```python
def cache(filename: str, mode: str | None = "a+") -> Middleware:
    """Set the cache filename for this model."""
def fresh() -> Middleware:
    """require any cached reply be ignored, and a fresh reply be generated."""
```
## Generalized Middleware


```python
class MetaModel(BaseModel):
    system: str | None

    @abstractmethod
    def chat(self, prompt, next: LanguageModel) -> Reply:
        """Promote a chat-with-prompt into a follow-on call of the next model."""

def meta(model: MetaModel) -> Middleware:
    """provide a meta model as middleware"""
```

See [meta model](examples/meta_model/README.md) for a full example. The `meta` 
middleware is really powerful and general, and can be used to build
models that use compute to generate useful answers.

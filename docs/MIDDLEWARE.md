

### Middleware

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
connect("mistral") | cache("cache.db") | echo() 
#  LLM ----> REPLY GOES LEFT TO RIGHT ------> Response
```

In the first example, we check the cache for any existing answer, then call echo
only if we do not have a cache hit. In the second, we always echo, even if the
cache hits. The user has the capability to choose the middleware stack they want
for their application.

Haverscript provides following middleware

| Middleware | Purpose |
|------------|---------|
| Retry      | retry on failure (using tenacity) |
| Validation | Fail under given condition |
| Echo       | Print prompt and reply |
| Stats      | Print basic stats about LLM |
| Cache      | Store and/or query prompt-reply pairs in DB |
| Transcript | Store a complete transcript of every call |
| Fresh      | Request a fresh reply (not cached) |
| Model      | Request a specific model be used |
| Options    | Set specific LLM options (such as seed) |
| Meta       | Support for generalized prompt and response transformation |

For examples, see

* [System prompt](examples/tree_of_calls/README.md) in tree of calls,
* [enabling the cache](examples/cache/README.md), 
* [JSON output](examples/check/README.md) in checking output, and
* [setting ollama options](examples/options/README.md).

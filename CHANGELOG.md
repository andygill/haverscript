# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2024-XX-YY
### Added
- Adding `Middleware` type for composable prompt and response handlers.
- `Middleware` can be added using `|`, giving a small pipe-based representation of flow.
  We have the following middleware components:

  - `echo()` adds eching of prompts and replies.
  - `retry()` which uses the tenacity package to provide a generic retry.
  - `validate()` which checks the response for a predicate.
  - `stats()` adds a dynamic single line summary of each LLM call.
  - `cache()` add a caching component.
  - `transcript()` adds a transcript component (transcripts the session to a file).
  - `options()` sets specific options.
  - `model()` set the model being used.
  
- Adding prompt specific flags to `Model.chat`.
  - `format: str | dict` is a request for a specially formated output.
  - `images : list[str]` are images to be passed to the model.
  - `fresh: bool` is a request to generate a new response.
  - `middleware: Middleware` appends a chat-specific middleware to the call.
  - `raw: bool` turns off prompt indentation cleanup (rarely needed)
- Added `Service` class, that can be asked about models, and can generate `Model`s.
- Added `response.value`, which return the JSON `dict` of the reply (or `None`).
- Added new `host` argument to `connect`, which allows for user "virtual" models.
- Added spinner when waiting for the first token from LLM when using `echo`.
- Added `metrics` to `Response`, which contains basic metrics about the LLM call.
- Added `render()` method to `Model`, for outputing markdown-style session viewing.
- Added `load()` method to `Model`, for parsing markdown-style sessions.
- Added `LLMError`, and subclasses. 
- Added `reject()` to `Response`, which raises a `LLMResultError` exception.
- Added support for together.ai's API as a first-class alternative to ollama.
- Added many more examples.
### Fixed
### Changed
- Updated `children` method to return all children when no prompt is supplied.
- Reworked SQL cache schema to store context as chain of responses, and use a
  string pool.
- Using the cache now uses LLM results in order, until exhausted, then calls the LLM.
### Removed
There are some breaking API changes. In all cases, the functionality has been
replaced with something more general and principled.

The concepts that caused changes are
- One you have a `Response`, that interaction with the LLM is considered done.
  There are no longer functions that attempt to re-run the call. Instead, middleware
  functions can be used to filter out responses as needed.
- The is not longer the concept of a `Response` being "fresh". Instead, the
  cache uses a cursor when reading cached responses, and it is possible to ask 
  that a specific interaction bypasses the cache.
- Formated output is a property of the specific call to chat, not the session.
- Most helper methods (`echo()`, `cache()`, etc) are now Middleware, and thus
  more flexable.

Specifically, here are the changes:
- Removed `check()` and `redo()` from `Response`.
  Replace it with `validate()` and `retry()` *before* the call to chat,
  or as chat-specific middleware.
- Removed `fresh` from `Response`. The concept of fresh responses has been replaced
  with a more robust caching middleware. There is now `fresh : bool = ...`
  argument to `Model.chat`, if a fresh output is needed.
- Removed `json()` from `Model`. It is replaced with the more general
  `format = "json"` as an argument to `Model.chat`.
- `echo()` and `cache()` are no longer `Model` methods, and now `Middleware` instances.

So, previously we would have `sesssion = connect("modelname").echo()`, and we now have
`sesssion = connect("modelname") | echo()`.



## [0.1.0] - 2024-09-23
### Initial release
- First release of the project.

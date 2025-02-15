# Changelog

Haverscript is a language for building LLM-based agents, providing a flexible
and composable way to interact with large language models. This CHANGELOG
documents notable changes to the Haverscript project.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.2] - 2025-??-??
### Added
- Initial agentic support. An `Agent` is a python class that has access to an LLM.
- `Markdown`, a simple DSL for building markdown-style prompts.
- `tools` argument for `Model.chat`, that supplies function-calling callbacks.
- `format` now supports standard types such as `list[int]` and `bool` for stuctured output.
- `echo` middleware now takes an optional `stream` parameter, defaulting to `True`.
- `Model.compress()` can remove older chat calls from the history, reducing context sizes.
- `connect_chatbot` which promotes a chatbot into a `Model`.
### Deprecated
- `dedent` has stubed out (has no effect). This has been replaced by Markdown support.
### Changed
- Internally, resolve refs from JSON schemas in format (this works around an ollama bug)
- `Model` and `Response` are now pydantic classes.

## [0.2.1] - 2024-12-30
### Added
- Support for Python 3.10 and 3.11. Python 3.9 and earlier is not supported.

## [0.2.0] - 2024-12-30
### Added
- Adding `Middleware` type for composable prompt and response handlers.
- `Middleware` can be added using `|`, giving a small pipe-based representation of flow.
  The following middleware components are available:

  - `echo()` adds echoing of prompts and replies.
  - `retry()` which uses the tenacity package to provide a generic retry.
  - `validate()` which checks the response for a predicate.
  - `stats()` adds a dynamic single line summary of each LLM call.
  - `cache()` add a caching component.
  - `transcript()` adds a transcript component (transcripts the session to a file).
  - `trace()` logs the calls through the middleware in both directions.
  - `fresh()` requests a fresh call to the LLM.
  - `options()` sets specific options.
  - `model()` set the model being used.
  - `format()` requires the output in JSON, with an optional pydantic class schema.
  - `meta()` is a hook to allow middleware to act like a test-time LLM.

- Adding prompt specific flags to `Model.chat`.
  - `images : list[str]` are images to be passed to the model.
  - `middleware: Middleware` appends a chat-specific middleware to the call.
- Added `Service` class, that can be asked about models, and can generate `Model`s.
- Added `response.value`, which return the JSON `dict` of the reply, the pydantic class, or `None`.
- Added spinner when waiting for the first token from LLM when using `echo`.
- Added `metrics` to `Response`, which contains basic metrics about the LLM call.
- Added `render()` method to `Model`, for outputing markdown-style session viewing.
- Added `load()` method to `Model`, for parsing markdown-style sessions.
- Added `LLMError`, and subclasses. 
- Added support for together.ai's API as a first-class alternative to ollama.
- Added many more examples.
- Added many more tests.
### Fixed
### Changed
- Updated `children` method to return all children when no prompt is supplied.
- Reworked SQL cache schema to store context as chain of responses, and use a
  string pool.
- Using the cache now uses LLM results in order, until exhausted, then calls the LLM.
### Removed
This release includes breaking API changes, which are outlined below. In all
cases, the functionality has been replaced with something more general and
principled.

The concepts that caused changes are
- One you have a `Response`, that interaction with the LLM is considered done.
  There are no longer functions that attempt to re-run the call. Instead, middleware
  functions can be used to filter out responses as needed.
- The is not longer the concept of a `Response` being "fresh". Instead, the
  cache uses a cursor when reading cached responses, and it is possible to ask 
  that a specific interaction bypasses the cache (using the `fresh()` middleware).
- Most helper methods (`echo()`, `cache()`, etc) are now Middleware, and thus
  more flexible.

Specifically, here are the changes:
- Removed `check()` and `redo()` from `Response`.
  Replace it with `validate()` and `retry()` *before* the call to chat,
  or as chat-specific middleware.
- Removed `fresh` from `Response`. The concept of fresh responses has been replaced
  with a more robust caching middleware. There is now `fresh()` middleware.
- Removed `json()` from `Model`. It is replaced with the more general
  `format()` middleware.
- `echo()` and `cache()` are no longer `Model` methods, and now `Middleware` instances.
- The utility functions `accept` and `valid_json` are removed.  They added no value,
  given the removal of `redo`.

So, previously we would have `session = connect("modelname").echo()`, and we now have
`session = connect("modelname") | echo()`.


## [0.1.0] - 2024-09-23
### Initial release
- First release of the project.

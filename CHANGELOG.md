# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2024-XX-YY
### Added
- Added `middleware(...)` method to `Model`, to support composable
  language models components. We have the following middleware components:
  - `Model.echo()` adds eching of prompts and replies.
  - `Model.retry()` which uses the tenacity package to provide a generic retry.
  - `Model.validate()` which checks the response for a predicate.
- Added `Service` class, that can be asked about models, and can generate `Model`s.
- Added `image(...)` method to `Model`, for multi-modal models.
- Added `response.value`, which return the JSON `dict` of the reply, or `None`.
- Added new `host` argument to `connect`, which allows for user "virtual" models.
- Added top-level `list_models` function to list model options available.
- Added spinner when waiting for the first token from LLM.
- Added `metrics` to `Response`, which contains basic metrics about the LLM call.
- Added `render()` method to `Model`, for outputing markdown-style session viewing.
- Added `load()` method to `Model`, for parsing markdown-style sessions.
- Added LLMError, and subclasses. 
- Added `reject()` to `Response`, which raises a `LLMResultError` exception.
- Added support for together.ai's API as a first-class alternative to ollama.
### Fixed
### Changed
- Updated `children` method to return all children when no prompt is supplied.
- Reworked SQL cache schema to store context as chain of responses, and use a
  string pool.
- Using the cache now uses LLM results in order, until exhausted, then calls the LLM.
### Removed
- Removed `check()` from `Response`. Replace it with `validate()` *before* the call to chat.
  The idea is tha `validate` can be paired with `retry` to build a custom repeater,
  and replaces the ad-hoc `check` which was doing redo's. Use middleware to build
  a LLM stack that works for you, not fix things afterwards.


## [0.1.0] - 2024-09-23
### Initial release
- First release of the project.

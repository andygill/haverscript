# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2024-XX-YY
### Added
- Generalized `echo()` to accept an `Echo` class.
- Added `image(...)` method to `Model`, for multi-modal models.
- Added `response.value`, which return the JSON `dict` of the reply, or `None`.
- Added new `host` argument to `connect`, which allows for user "virtual" models.
- Added top-level `list_models` function to list model options available.
- Added spinner when waiting for the first token from LLM.
### Fixed
### Changed
- Updated `children` method to return all children when no prompt is supplied.
### Removed


## [0.1.0] - 2024-09-23
### Initial release
- First release of the project.

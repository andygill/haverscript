class LLMError(Exception):
    """Base exception for all LLM-related errors."""


class LLMConfigurationError(LLMError):
    """Exception raised for errors occurring during LLM configuration."""


class LLMRequestError(LLMError):
    """Exception raised for errors occurring during the request to the LLM."""


class LLMConnectivityError(LLMRequestError):
    """Exception raised due to connectivity issues with the LLM service."""


class LLMPermissionError(LLMRequestError):
    """Exception raised when access to the LLM is denied due to permission issues."""


class LLMRateLimitError(LLMRequestError):
    """Exception raised when the rate limit is exceeded with the LLM service."""


class LLMResponseError(LLMError):
    """Exception raised for errors occurring during the response from the LLM."""


class LLMResultError(LLMError):
    """Exception raised for errors related to the LLM's output quality."""


class LLMInternalError(LLMError):
    """Exception raised when something was inconsistent internally"""

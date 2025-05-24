# File: src/lean_explore/llm_caller.py

"""Provides a client for interacting with Google's Gemini API.

This module defines classes and functions to facilitate communication with
Google's Gemini large language models (LLMs) for both text generation and
embedding creation. It includes features like asynchronous API calls, automatic
retries with exponential backoff for transient errors, and integrated cost
tracking based on model usage (tokens/units).
"""

import asyncio
import os
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

# Google GenAI Library
import google.generativeai as genai
from google.api_core import (
    exceptions as google_api_exceptions,
)  # For specific API errors
from google.generativeai import types as genai_types

try:
    # Import both the config dictionary and the specific API key getter
    from .config import APP_CONFIG, get_gemini_api_key
except ImportError:
    # Updated warning message to reflect the new path
    warnings.warn(
        "config_loader.APP_CONFIG and get_gemini_api_key not found."
        " Using fallbacks and environment variables directly.",
        ImportWarning,
    )
    # Provide fallbacks if the config loader isn't available
    APP_CONFIG = {}

    # Define a fallback getter if the real one isn't imported
    def get_gemini_api_key() -> Optional[str]:
        """Fallback: Retrieves API key directly from environment."""
        return os.getenv("GEMINI_API_KEY")


# These can serve as hardcoded fallbacks if environment variables AND arguments
# are missing/invalid
FALLBACK_MAX_RETRIES = 3
FALLBACK_BACKOFF_FACTOR = 1.0
# Default if env var/config is missing - should match config.yml ideally
FALLBACK_EMBEDDING_MODEL = "models/text-embedding-004"

# Default safety settings - defined here for use as default in __init__
# Ensure genai_types is available before creating the list
DEFAULT_SAFETY_SETTINGS = (
    [
        genai_types.SafetySettingDict(
            category=genai_types.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=genai_types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        ),
        genai_types.SafetySettingDict(
            category=genai_types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=genai_types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        ),
        genai_types.SafetySettingDict(
            category=genai_types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=genai_types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        ),
        genai_types.SafetySettingDict(
            category=genai_types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=genai_types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        ),
    ]
    if genai_types
    else None
)

# --- Cost Tracking ---


@dataclass
class ModelUsageStats:
    """Stores usage statistics for a specific model.

    Attributes:
        calls (int): Number of successful API calls made for the model.
        prompt_tokens (int): Total number of input tokens/units processed by
            the model.
        completion_tokens (int): Total number of output tokens/units generated
            by the model (often 0 for embeddings).
    """

    calls: int = 0
    prompt_tokens: int = 0  # Represents input tokens/units
    completion_tokens: int = (
        0  # Represents output tokens/units (often 0 for embeddings)
    )


@dataclass
class ModelCostInfo:
    """Stores cost per MILLION units (tokens/chars) for a specific model.

    Attributes:
        input_cost_per_million_units (float): The cost for processing one
            million input units (e.g., tokens).
        output_cost_per_million_units (float): The cost for generating one
            million output units (e.g., tokens).
    """

    input_cost_per_million_units: float
    output_cost_per_million_units: float


class GeminiCostTracker:
    """Tracks API usage and estimates costs for Google Gemini models.

    This class maintains counts of API calls, input/output units (tokens),
    and calculates estimated costs based on predefined rates per million units.
    It handles both generative and embedding models, sourcing cost data from
    a JSON configuration.

    Attributes:
        _usage_stats (Dict[str, ModelUsageStats]): A dictionary mapping model
            names to their accumulated usage statistics.
        _model_costs (Dict[str, ModelCostInfo]): A dictionary mapping model
            names to their cost information (per million units).
    """

    def __init__(self, model_costs_override: Optional[Dict[str, Any]] = None):
        """Initializes the GeminiCostTracker.

        Loads model cost information primarily from APP_CONFIG['costs']. An
        optional override dictionary can be provided. Initializes internal
        dictionaries to store usage and cost data.

        Args:
            model_costs_override (Optional[Dict[str, Any]]): A dictionary
                mapping model names (e.g., "gemini-1.5-flash-latest") to
                their costs per million units, in the format
                `{"model_name": {"input": float, "output": float}, ...}`.
                This overrides costs loaded from APP_CONFIG.
        """
        # Prioritize override, then APP_CONFIG, then empty dict
        effective_costs_dict = (
            model_costs_override
            if model_costs_override is not None
            else APP_CONFIG.get("costs", {})
        )  # Get costs dict from APP_CONFIG
        self._usage_stats: Dict[str, ModelUsageStats] = {}
        self._model_costs: Dict[str, ModelCostInfo] = {}
        self._parse_model_costs(effective_costs_dict)  # Pass the dictionary directly

    def _parse_model_costs(self, costs_dict: Dict[str, Any]):
        """Parses the model costs dictionary.

        Populates the `_model_costs` dictionary with ModelCostInfo objects
        from the provided dictionary (typically from APP_CONFIG['costs']).
        Handles potential invalid formats gracefully, issuing warnings.
        Assumes costs are provided per million units.

        Args:
            costs_dict (Dict[str, Any]): The dictionary containing model cost data.
        """
        if not isinstance(costs_dict, dict):
            warnings.warn(
                "Invalid format for costs data: Expected dict, got "
                f"{type(costs_dict)}. Costs will not be tracked."
            )
            return
        try:
            # Iterate directly over the dictionary provided
            for model, costs in costs_dict.items():
                # Allow 'output' to be missing or 0 for embedding models
                if isinstance(costs, dict) and "input" in costs:
                    input_cost = float(costs["input"])
                    # Default output cost to 0 if not specified
                    output_cost = float(costs.get("output", 0.0))
                    self._model_costs[model] = ModelCostInfo(
                        input_cost_per_million_units=input_cost,
                        output_cost_per_million_units=output_cost,
                    )
                else:
                    warnings.warn(
                        f"Invalid cost format for model '{model}' in costs data. "
                        f"Expected at least {{'input': float}}. Found: {costs}"
                    )
        # No JSONDecodeError needed, but catch other potential issues
        except (TypeError, ValueError) as e:
            warnings.warn(
                f"Error processing cost data entry: {e}. Check cost configuration "
                "format."
            )
        except Exception as e:
            warnings.warn(f"Unexpected error processing costs data: {e}")

    def record_usage(self, model: str, input_units: int, output_units: int):
        """Records usage statistics for a specific model after a successful API call.

        Increments the call count and adds the input and output units (tokens)
        to the totals for the given model name.

        Args:
            model (str): The name of the model used in the API call.
            input_units (int): The number of input units (e.g., tokens) consumed.
            output_units (int): The number of output units (e.g., tokens) generated.
        """
        if model not in self._usage_stats:
            self._usage_stats[model] = ModelUsageStats()

        stats = self._usage_stats[model]
        stats.calls += 1
        stats.prompt_tokens += input_units  # Map input_units -> prompt_tokens
        stats.completion_tokens += output_units  # Map output_units -> completion_tokens

    def get_total_cost(self) -> float:
        """Calculates the estimated total cost across all tracked models.

        Sums the costs for each model based on its recorded usage (input/output
        units) and the cost information loaded during initialization (cost per
        million units). Issues warnings for models with recorded usage but
        missing cost data.

        Returns:
            float: The total estimated cost in the currency defined by the
            cost data (e.g., USD).
        """
        total_cost = 0.0
        for model, stats in self._usage_stats.items():
            if model in self._model_costs:
                costs = self._model_costs[model]
                total_cost += (
                    stats.prompt_tokens / 1_000_000.0
                ) * costs.input_cost_per_million_units + (
                    stats.completion_tokens / 1_000_000.0
                ) * costs.output_cost_per_million_units
            else:
                warnings.warn(
                    f"Cost information missing for model '{model}'. Usage for this "
                    "model is not included in total cost."
                )
        return total_cost

    def get_summary(self) -> Dict[str, Any]:
        """Generates a summary report of API usage and estimated costs.

        Provides a dictionary containing the overall estimated cost and a
        breakdown of usage (calls, input/output units) and estimated cost
        per model.

        Returns:
            Dict[str, Any]: A dictionary summarizing usage and costs, e.g.:
            ```
            {
                "total_estimated_cost": 1.23,
                "usage_by_model": {
                    "gemini-1.5-flash-latest": {
                        "calls": 10,
                        "input_units": 50000,
                        "output_units": 10000,
                        "estimated_cost": 0.85
                    },
                    "models/text-embedding-004": {
                        "calls": 5,
                        "input_units": 200000,
                        "output_units": 0,
                        "estimated_cost": 0.38
                    }
                }
            }
            ```
            If cost data is missing for a model, its "estimated_cost" will
            be "Unknown (cost data missing)".
        """
        total_estimated_cost = (
            self.get_total_cost()
        )  # Ensure warnings are potentially triggered
        summary: Dict[str, Any] = {
            "total_estimated_cost": total_estimated_cost,
            "usage_by_model": {},
        }
        for model, stats in self._usage_stats.items():
            model_summary = {
                "calls": stats.calls,
                "input_units": stats.prompt_tokens,
                "output_units": stats.completion_tokens,  # Usually 0 for embeddings
                "estimated_cost": 0.0,
            }
            if model in self._model_costs:
                costs = self._model_costs[model]
                model_summary["estimated_cost"] = (
                    stats.prompt_tokens / 1_000_000.0
                ) * costs.input_cost_per_million_units + (
                    stats.completion_tokens / 1_000_000.0
                ) * costs.output_cost_per_million_units
            else:
                model_summary["estimated_cost"] = "Unknown (cost data missing)"

            summary["usage_by_model"][model] = model_summary
        return summary


# --- Gemini Client ---


class GeminiClient:
    """Client for Google Gemini API with retries and cost tracking.

    Provides asynchronous methods (`generate`, `embed_content`) to interact
    with Google's generative and embedding models. Includes automatic retries
    on transient errors (like rate limits or server issues) with exponential
    backoff. Integrates with `GeminiCostTracker` to monitor API usage and
    estimate costs. Configuration is loaded from APP_CONFIG (via config_loader)
    or passed arguments.

    Attributes:
        api_key (str): The Google AI API key being used.
        default_generation_model (str): The default model used for `generate` calls.
        default_embedding_model (str): The default model used for `embed_content` calls.
        max_retries (int): The maximum number of retry attempts for failed API calls.
        backoff_factor (float): The base factor for exponential backoff delays between
            retries.
        cost_tracker (GeminiCostTracker): The instance used for tracking usage and
            costs.
        safety_settings (Optional[list]): Default safety settings applied to
            `generate` calls.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_generation_model: Optional[str] = None,
        default_embedding_model: Optional[str] = None,
        max_retries: Optional[int] = None,
        backoff_factor: Optional[float] = None,
        cost_tracker: Optional[GeminiCostTracker] = None,
        safety_settings: Optional[list] = DEFAULT_SAFETY_SETTINGS,
    ):
        """Initializes the Gemini Client.

        Configures the client using provided arguments, falling back to values
        from APP_CONFIG (loaded by config_loader.py) or hardcoded defaults.
        Sets up the API key, default models, retry parameters,
        cost tracker, and safety settings. Configures the underlying
        `google.generativeai` library.

        Args:
            api_key (Optional[str]): Google AI API key. If None, reads from
                `GEMINI_API_KEY` environment variable via `get_gemini_api_key`.
            default_generation_model (Optional[str]): Default model name for
                generation (e.g., "gemini-2.0-flash"). If None, reads from
                `APP_CONFIG['llm']['generation_model']`.
            default_embedding_model (Optional[str]): Default model name for
                embeddings (e.g., "models/text-embedding-004"). If None, reads
                from `APP_CONFIG['llm']['embedding_model']` or uses
                `FALLBACK_EMBEDDING_MODEL`. Ensures 'models/' prefix.
            max_retries (Optional[int]): Maximum retry attempts for API calls.
                If None, reads from `APP_CONFIG['llm']['retries']` or uses
                `FALLBACK_MAX_RETRIES`. Must be non-negative.
            backoff_factor (Optional[float]): Base factor for exponential backoff
                delay (seconds). If None, reads from `APP_CONFIG['llm']['backoff']`
                or uses `FALLBACK_BACKOFF_FACTOR`. Must be non-negative.
            cost_tracker (Optional[GeminiCostTracker]): An instance of
                `GeminiCostTracker` to record usage. If None, a new instance
                is created internally using costs from APP_CONFIG.
            safety_settings (Optional[list]): Default safety settings for
                generation, as a list of `genai_types.SafetySettingDict`.
                Defaults to `DEFAULT_SAFETY_SETTINGS`.

        Raises:
            RuntimeError: If the `google.generativeai` package is not installed
                or if configuring the underlying client fails.
            ValueError: If the API key or default generation model is missing
                (and not found via config loader or environment variables).
        """
        if not genai:
            raise RuntimeError("google.generativeai package is required but not found.")

        # --- Configuration Loading ---
        self.api_key = api_key or get_gemini_api_key()
        if not self.api_key:
            raise ValueError(
                "Gemini API key is missing. Set via argument, GEMINI_API_KEY "
                "environment variable, or ensure config loader works."
            )

        # Get from arg, then APP_CONFIG['llm']['generation_model']
        _config_gen_model = APP_CONFIG.get("llm", {}).get("generation_model")
        self.default_generation_model = default_generation_model or _config_gen_model
        if not self.default_generation_model:
            raise ValueError(
                "Default Gemini generation model is missing. Set via argument "
                "or in config file ('llm.generation_model')."
            )

        # Get from arg, then APP_CONFIG['llm']['embedding_model'],
        # then fallback constant
        _config_emb_model = APP_CONFIG.get("llm", {}).get("embedding_model")
        _emb_model_name = default_embedding_model or _config_emb_model

        if not _emb_model_name:
            warnings.warn(
                f"Default embedding model not set via argument or "
                "config ('llm.embedding_model'). "
                f"Using fallback: {FALLBACK_EMBEDDING_MODEL}"
            )
            self.default_embedding_model = FALLBACK_EMBEDDING_MODEL
        else:
            # Ensure the model name starts with 'models/' for consistency if it
            # doesn't already
            if not _emb_model_name.startswith("models/"):
                self.default_embedding_model = f"models/{_emb_model_name}"
                warnings.warn(
                    f"Resolved embedding model '{_emb_model_name}' did not start with "
                    f"'models/'. Using '{self.default_embedding_model}' for "
                    "consistency."
                )
            else:
                self.default_embedding_model = _emb_model_name

        # Max Retries
        # Get from arg, then APP_CONFIG['llm']['retries'], then fallback constant
        _config_retries = APP_CONFIG.get("llm", {}).get("retries")
        # Check if config value is valid, otherwise use fallback
        if isinstance(_config_retries, int) and _config_retries >= 0:
            _effective_retries = _config_retries
        else:
            _effective_retries = FALLBACK_MAX_RETRIES
            if (
                _config_retries is not None
            ):  # Warn if config value was present but invalid
                warnings.warn(
                    f"Invalid 'llm.retries' value in config: "
                    f"'{_config_retries}'. Using default {FALLBACK_MAX_RETRIES}."
                )

        # Argument overrides config/default
        self.max_retries = (
            max_retries if max_retries is not None else _effective_retries
        )
        self.max_retries = max(0, self.max_retries)  # Ensure non-negative

        # Backoff Factor
        # Get from arg, then APP_CONFIG['llm']['backoff'], then fallback constant
        _config_backoff = APP_CONFIG.get("llm", {}).get("backoff")
        # Check if config value is valid, otherwise use fallback
        if isinstance(_config_backoff, (float, int)) and _config_backoff >= 0:
            _effective_backoff = float(_config_backoff)
        else:
            _effective_backoff = FALLBACK_BACKOFF_FACTOR
            if (
                _config_backoff is not None
            ):  # Warn if config value was present but invalid
                warnings.warn(
                    f"Invalid 'llm.backoff' value in config: "
                    f"'{_config_backoff}'. Using default {FALLBACK_BACKOFF_FACTOR}."
                )

        # Argument overrides config/default
        self.backoff_factor = (
            backoff_factor if backoff_factor is not None else _effective_backoff
        )
        self.backoff_factor = max(0.0, self.backoff_factor)  # Ensure non-negative

        # --- Initialization ---
        # Pass costs from APP_CONFIG to the tracker if no override is provided
        self.cost_tracker = (
            cost_tracker
            if cost_tracker is not None
            else GeminiCostTracker(model_costs_override=APP_CONFIG.get("costs"))
        )
        self.safety_settings = safety_settings  # Used by generate method

        try:
            genai.configure(api_key=self.api_key)
        except Exception as e:
            raise RuntimeError(f"Failed to configure Google GenAI client: {e}") from e

    # --- Private Helper for Retries ---
    async def _execute_with_retry(
        self,
        api_call_func: Callable[..., Any],
        *args: Any,
        _model_name_for_log: str = "unknown_model",
        **kwargs: Any,
    ) -> Any:
        """Executes a synchronous API call asynchronously with retry logic.

        Wraps a synchronous Google GenAI SDK function call (like
        `model.generate_content` or `genai.embed_content`) using
        `asyncio.to_thread`. Implements exponential backoff retries for specific
        Google API errors (rate limits, server errors) and general exceptions.

        Args:
            api_call_func (Callable[..., Any]): The synchronous Google GenAI SDK
                function to call (e.g., `model_instance.generate_content`).
            *args: Positional arguments to pass to `api_call_func`.
            _model_name_for_log (str): The name of the model being called, used
                for logging/warning messages.
            **kwargs: Keyword arguments to pass to `api_call_func`.

        Returns:
            Any: The result returned by the successful `api_call_func`.

        Raises:
            google_api_exceptions.GoogleAPIError: If a non-retryable API error
                (like 4xx client errors) occurs.
            Exception: If the API call fails after all retry attempts due to
                retryable API errors or other exceptions. The specific exception
                encountered on the last attempt is raised.
        """
        final_error: Optional[Exception] = None
        model_name = _model_name_for_log

        total_attempts = self.max_retries + 1
        for attempt in range(total_attempts):
            try:
                # Use asyncio.to_thread to run the synchronous SDK call in a
                # separate thread
                # Note: This assumes api_call_func itself is synchronous.
                response = await asyncio.to_thread(api_call_func, *args, **kwargs)
                return response  # Success

            except google_api_exceptions.ResourceExhausted as e:
                # Specific handling for rate limits / quota errors - retryable
                final_error = e
                warnings.warn(
                    f"API Quota/Rate Limit Error for {model_name} on attempt "
                    f"{attempt + 1}/{total_attempts}: {e}"
                )
                # Continue to retry logic below

            except google_api_exceptions.GoogleAPIError as e:
                # Catch other Google API errors (e.g., server errors, bad requests)
                final_error = e
                # Decide if retryable based on status code maybe? For now, retry most.
                # 4xx errors are typically not retryable (Bad Request, Not Found,
                # Invalid Argument)
                # Allow 429 (ResourceExhausted, handled above) to be retryable.
                status_code = getattr(e, "code", 0)
                if 400 <= status_code < 500 and status_code != 429:
                    warnings.warn(
                        f"API Client Error (4xx) for {model_name} on attempt "
                        f"{attempt + 1}/{total_attempts}: {e}. Not retrying."
                    )
                    break  # Don't retry most client errors
                else:  # Retry server errors (5xx), 429, or unknown API errors
                    warnings.warn(
                        f"API Server/Retryable Error for {model_name} on attempt "
                        f"{attempt + 1}/{total_attempts}: {e}"
                    )
                # Continue to retry logic below

            except Exception as e:
                # Catch broader exceptions (network issues, unexpected errors during
                # async wrapper)
                final_error = e
                warnings.warn(
                    f"Unexpected Error during API call for {model_name} on attempt "
                    f"{attempt + 1}/{total_attempts}: {e}"
                )
                # Continue to retry logic below

            # --- Retry Logic ---
            if attempt < self.max_retries:
                sleep_time = self.backoff_factor * (2**attempt)
                retries_remaining = self.max_retries - attempt
                warnings.warn(
                    f"Retrying API call for {model_name} in {sleep_time:.2f} "
                    f"seconds... ({retries_remaining} retries remaining)"
                )
                await asyncio.sleep(sleep_time)
            else:
                # This was the final attempt
                warnings.warn(
                    f"API call for {model_name} failed on the final attempt "
                    f"({attempt + 1}/{total_attempts})."
                )
                break  # Exit loop after final attempt

        # If loop finished without returning, raise the last captured error
        if final_error is not None:
            raise final_error
        else:
            # This case should ideally not be reached if the loop logic is correct
            raise Exception(
                f"Unknown error during API call to {model_name} after "
                f"{total_attempts} attempts"
            )

    # --- Public API Methods ---

    async def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        generation_config_override: Optional[Dict[str, Any]] = None,
        safety_settings_override: Optional[list] = None,
    ) -> str:
        """Generates content using a specified Gemini model with retry logic.

        Constructs the request with the prompt and optional system instruction,
        generation config, and safety settings. Uses the `_execute_with_retry`
        helper to handle the API call to the generative model. Processes the
        response to extract the text content and records usage statistics if a
        cost tracker is configured and usage metadata is available.

        Args:
            prompt (str): The main user prompt for content generation.
            model (Optional[str]): The specific Gemini model name to use
                (e.g., "gemini-1.5-flash-latest"). If None, uses the client's
                `default_generation_model`.
            system_prompt (Optional[str]): An optional system instruction to guide
                the model's behavior. Passed during model initialization.
            generation_config_override (Optional[Dict[str, Any]]): A dictionary
                containing generation parameters (like temperature, top_p,
                max_output_tokens) to override the model's defaults. See
                `google.generativeai.types.GenerationConfig` for options.
            safety_settings_override (Optional[list]): A list of safety setting
                dictionaries to override the client's default safety settings
                for this specific call. See
                `google.generativeai.types.SafetySettingDict`.

        Returns:
            str: The generated text content from the model.

        Raises:
            ValueError: If the specified model name is invalid, if the API
                response indicates the content was blocked due to safety
                settings, or if the response structure is unexpected or lacks
                text content.
            Exception: If the API call fails after all retry attempts (reraised
                from `_execute_with_retry`).
        """
        effective_model = model or self.default_generation_model
        gen_config = (
            genai_types.GenerationConfig(**generation_config_override)
            if generation_config_override
            else None
        )
        safety_settings = (
            safety_settings_override
            if safety_settings_override is not None
            else self.safety_settings
        )

        try:
            # Initialize model instance - validation happens here
            # Note: System instruction should be passed here if supported by the
            # specific model version/SDK
            model_instance = genai.GenerativeModel(
                effective_model,
                system_instruction=system_prompt,  # Pass system prompt here
            )
        except Exception as e:
            # Catch errors during model initialization (e.g., invalid name)
            raise ValueError(
                f"Failed to initialize generative model '{effective_model}'. "
                f"Check model name. Error: {e}"
            ) from e

        contents = [{"role": "user", "parts": [prompt]}]
        # System prompt is handled by model instance initialization now

        try:
            # Use the retry helper
            api_kwargs = {
                "contents": contents,
                "generation_config": gen_config,
                "safety_settings": safety_settings,
            }
            # api_call_func is the bound method model_instance.generate_content
            response = await self._execute_with_retry(
                model_instance.generate_content,
                _model_name_for_log=effective_model,  # Log arg
                **api_kwargs,  # API args for generate_content
            )

            # --- Process Response ---
            generated_text = None
            prompt_tokens = 0
            completion_tokens = 0
            usage_metadata = getattr(response, "usage_metadata", None)

            try:
                # Attempt to access generated text safely
                # response.text can raise ValueError if content is blocked
                generated_text = response.text
            except ValueError as e:
                # Handle cases where accessing .text fails (e.g., blocked content)
                block_reason = "Unknown"
                try:
                    # Try to extract the blocking reason from prompt_feedback
                    if (
                        response.prompt_feedback
                        and response.prompt_feedback.block_reason
                    ):
                        # Use the enum name if possible, otherwise the string
                        # representation
                        block_reason = getattr(
                            response.prompt_feedback.block_reason,
                            "name",
                            str(response.prompt_feedback.block_reason),
                        )
                except AttributeError:
                    pass  # Ignore if prompt_feedback structure is different
                raise ValueError(
                    f"API call failed for {effective_model}: Content blocked or "
                    f"invalid. Reason: {block_reason}. Original Error: {e}"
                ) from e
            except AttributeError:
                # If .text attribute doesn't exist (shouldn't happen with valid
                # response)
                pass  # We handle None generated_text below

            # Fallback text extraction if needed (though response.text should
            # usually work or raise error)
            if generated_text is None:
                try:
                    # Access text through candidates -> content -> parts structure
                    if (
                        response.candidates
                        and response.candidates[0].content
                        and response.candidates[0].content.parts
                    ):
                        generated_text = "".join(
                            part.text
                            for part in response.candidates[0].content.parts
                            if hasattr(part, "text")
                        )
                    # Ensure we actually got some text and it's not just whitespace
                    if not generated_text or not generated_text.strip():
                        raise ValueError(
                            f"API call failed for {effective_model}: Received no valid "
                            "text content in response structure."
                        )
                except (AttributeError, IndexError, ValueError) as text_extract_err:
                    # Raise if the structure traversal or final check fails
                    raise ValueError(
                        f"API call failed for {effective_model}: Could not extract "
                        "text from the expected response structure."
                    ) from text_extract_err

            # --- Token Counting & Cost Tracking ---
            if usage_metadata:
                try:
                    # Use attribute names consistent with google-genai v0.3+
                    prompt_tokens = getattr(usage_metadata, "prompt_token_count", 0)
                    # Sum of tokens across all generated candidates
                    completion_tokens = getattr(
                        usage_metadata, "candidates_token_count", 0
                    )
                    # Ensure they are integers
                    prompt_tokens = (
                        int(prompt_tokens) if prompt_tokens is not None else 0
                    )
                    completion_tokens = (
                        int(completion_tokens) if completion_tokens is not None else 0
                    )
                except (AttributeError, ValueError, TypeError) as e:
                    warnings.warn(
                        f"Error accessing token counts from usage metadata for "
                        f"{effective_model}: {e}. Cost tracking may be inaccurate."
                    )
                    prompt_tokens = 0
                    completion_tokens = 0

                if self.cost_tracker:
                    # Record usage: Map prompt_tokens -> input_units,
                    # completion_tokens -> output_units
                    self.cost_tracker.record_usage(
                        effective_model, prompt_tokens, completion_tokens
                    )
            else:
                warnings.warn(
                    f"Response object for model '{effective_model}' lacks "
                    f"'usage_metadata'. Cost tracking may be inaccurate."
                )

            # Ensure generated_text is not None before returning
            if generated_text is None:
                # This state should be highly unlikely given the checks above
                raise ValueError(
                    f"API call for {effective_model} resulted in "
                    "None text unexpectedly."
                )

            return generated_text

        except ValueError as ve:
            # Catch ValueErrors raised during response processing (blocking,
            # empty content)
            # These are considered definitive failures, don't wrap further
            raise ve
        except Exception as e:
            # Catch errors from _execute_with_retry (after all retries failed)
            # or potential errors during API response processing not caught above.
            # Avoid wrapping ValueError from above again
            if not isinstance(e, ValueError):
                raise Exception(
                    f"API call to generation model '{effective_model}' failed after "
                    "retries or during processing."
                ) from e
            else:
                raise e  # Re-raise the original ValueError

    async def embed_content(
        self,
        contents: Union[str, List[str]],
        task_type: str,
        *,
        model: Optional[str] = None,
        title: Optional[str] = None,
        output_dimensionality: Optional[int] = None,
    ) -> List[List[float]]:
        """Generates embeddings for text content(s) using a Gemini model.

        Handles both single string and batch (list of strings) embedding requests.
        Uses the `_execute_with_retry` helper for the API call. Processes the
        response dictionary to extract the embedding vector(s). Attempts to
        record usage statistics if a cost tracker is configured and usage
        metadata is available in the response (though often absent for embeddings).

        Args:
            contents (Union[str, List[str]]): A single string or a list of strings
                to be embedded.
            task_type (str): The intended task for the embedding, which influences
                its characteristics (e.g., "RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY",
                "SEMANTIC_SIMILARITY", "CLASSIFICATION"). Refer to Google AI
                documentation for valid task types.
            model (Optional[str]): The specific Gemini embedding model name to use
                (e.g., "models/text-embedding-004"). If None, uses the client's
                `default_embedding_model`. Should typically start with 'models/'.
            title (Optional[str]): An optional title for the document, used only
                when `task_type` is "RETRIEVAL_DOCUMENT". Ignored otherwise.
            output_dimensionality (Optional[int]): An optional integer to specify
                the desired dimension to truncate the resulting embedding vector(s) to.
                If None, the model's default dimensionality is used.

        Returns:
            List[List[float]]: A list containing the embedding vector(s). If a
            single string was provided as input, the outer list will contain
            a single vector (list of floats). If a list of strings was provided,
            the outer list will contain multiple vectors in the corresponding order.

        Raises:
            TypeError: If `contents` is not a string or a list of strings, or
                if it's a list containing non-string items.
            ValueError: If the API response is missing the expected 'embedding'
                or 'embeddings' key, or if the embedding format is invalid.
            Exception: If the API call fails after all retry attempts (reraised
                from `_execute_with_retry`).
        """
        effective_model = model or self.default_embedding_model
        if not effective_model.startswith("models/"):
            # Enforce 'models/' prefix based on how cost dict and API often work
            warnings.warn(
                f"Embedding model name '{effective_model}' should ideally start "
                "with 'models/'. Attempting call anyway."
            )
            # Consider adding prefix here if API consistently fails without it:
            # effective_model = f'models/{effective_model}'

        # Validate contents type
        if not isinstance(contents, (str, list)):
            raise TypeError("Input 'contents' must be a string or a list of strings.")
        if isinstance(contents, list) and not all(
            isinstance(item, str) for item in contents
        ):
            raise TypeError(
                "If 'contents' is provided as a list, all its items must be strings."
            )

        # Prepare arguments for genai.embed_content
        embed_args = {
            "model": effective_model,
            "content": contents,  # Pass str or list directly to the SDK function
            "task_type": task_type,
        }
        if title is not None and task_type == "RETRIEVAL_DOCUMENT":
            embed_args["title"] = title
        elif title is not None:
            # Warn if title is provided but task type doesn't support it
            warnings.warn(
                f"Ignoring 'title' argument as task_type is '{task_type}', not "
                "'RETRIEVAL_DOCUMENT'."
            )

        if output_dimensionality is not None:
            # Add output_dimensionality if provided
            embed_args["output_dimensionality"] = output_dimensionality

        try:
            # Use the retry helper; genai.embed_content is the sync function to wrap
            response_dict = await self._execute_with_retry(
                genai.embed_content,
                _model_name_for_log=effective_model,  # Log arg
                **embed_args,  # API args for embed_content
            )

            # --- Process Response ---
            embeddings: List[List[float]] = []

            if "embedding" in response_dict:
                result = response_dict["embedding"]
                # Check if the input was a list
                if isinstance(contents, list):
                    # Assume 'embedding' key holds the list of results for batch input
                    if isinstance(result, list) and all(
                        isinstance(e, list) for e in result
                    ):
                        # It's already the list we want: [[emb1], [emb2]]
                        embeddings = result
                    else:
                        # Handle potential malformed list response
                        raise ValueError(
                            f"Invalid embedding format received under 'embedding' key "
                            f"for list input from model {effective_model}."
                        )
                elif isinstance(result, list):
                    # Single input case: result is the vector (list of floats)
                    # Wrap the single embedding vector in a list to match return type
                    # List[List[float]]
                    embeddings = [result]  # result is [vector] -> return [[vector]]
                else:
                    # Handle potential malformed single response
                    raise ValueError(
                        f"Invalid embedding format received under 'embedding' key for "
                        f"single input from model {effective_model}."
                    )
            # Keep 'embeddings' check as fallback just
            # in case API behaviour changes/differs
            elif "embeddings" in response_dict:
                result = response_dict["embeddings"]
                if isinstance(result, list) and all(
                    isinstance(e, list) for e in result
                ):
                    embeddings = result
                else:
                    raise ValueError(
                        "Invalid embedding format received under 'embeddings' key "
                        f"from model {effective_model}."
                    )
            else:
                # Handle case where neither expected key is present
                raise ValueError(
                    f"API call for {effective_model} succeeded but the response "
                    "dictionary is missing the expected 'embedding' or 'embeddings' "
                    "key."
                )

            # --- Token Counting & Cost Tracking (Attempt) ---
            # NOTE: Usage metadata is often NOT included in embedding responses
            # from the API.
            # We attempt to extract it but expect it might be missing.
            usage_metadata = response_dict.get(
                "usage_metadata", None
            )  # Safely get metadata if present
            input_units = 0
            output_units = (
                0  # Embeddings only have input cost, output is fixed/implicit
            )

            if usage_metadata and isinstance(usage_metadata, dict):
                # Assume the key for input tokens might be 'total_token_count'
                # or similar
                # Adjust this key based on actual observed responses if possible.
                token_key = "total_token_count"  # Placeholder - check API response
                if token_key in usage_metadata:
                    try:
                        input_units = int(usage_metadata[token_key])
                    except (ValueError, TypeError):
                        warnings.warn(
                            f"Could not parse '{token_key}' from usage metadata for "
                            f"{effective_model}. Cost tracking might be inaccurate."
                        )
                        input_units = 0
                else:
                    warnings.warn(
                        f"Expected token count key ('{token_key}') not found in usage "
                        f"metadata for {effective_model}. Cost tracking might be "
                        "inaccurate."
                    )

                # Record usage only if we found a positive number of input units
                if input_units > 0 and self.cost_tracker:
                    self.cost_tracker.record_usage(
                        effective_model, input_units, output_units
                    )
                elif (
                    input_units == 0 and token_key in usage_metadata
                ):  # Only warn if key exists but value is bad/zero
                    # Only issue a warning if we found metadata but couldn't get
                    # tokens from it.
                    warnings.warn(
                        "Could not determine input units from available usage "
                        f"metadata for {effective_model}. Cost tracking may be "
                        "inaccurate for this call."
                    )

            else:
                # This is the common case: usage metadata is absent.
                # Issue a warning that cost tracking is skipped for this specific call.
                warnings.warn(
                    f"Usage metadata not found in response for embedding model "
                    f"'{effective_model}'. Cost tracking skipped for this call."
                )

            return embeddings

        except ValueError as ve:
            # Catch ValueErrors raised during response processing (e.g., invalid format)
            raise ve
        except Exception as e:
            # Catch errors from _execute_with_retry (after all retries failed)
            # or potential errors during API response processing not caught above.
            if not isinstance(e, ValueError):
                raise Exception(
                    f"API call to embedding model '{effective_model}' failed after "
                    "retries or during processing."
                ) from e
            else:
                raise e  # Re-raise the original ValueError

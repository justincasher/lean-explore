# File: src/lean_explore/config.py

"""Loads and manages application configuration from multiple sources.

Provides functions to load settings from YAML files (e.g., 'config.yml'
at project root), JSON files (e.g., 'model_costs.json' in a 'data/'
directory at project root), and environment variables (for overrides and
secrets). Paths for default configuration files are determined by:
1. Environment variables (LEAN_RANK_CONFIG_FILE, LEAN_RANK_COSTS_FILE).
2. Derived project root relative to this module's location (assuming
   this module is in 'src/lean_explore/config.py').
3. Fallback to current working directory (with a warning).

The loaded configuration is exposed via `APP_CONFIG`, a singleton dictionary.
Helper functions access sensitive values (e.g., API keys) from environment
variables.
"""

import json
import logging
import os
import pathlib
from typing import Any, Dict, List, Optional, Tuple, Type

import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
# If this module is imported and the root logger is already configured,
# this basicConfig call will not have an effect. If run standalone or
# no handlers are configured, this provides a default.
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: [%(name)s] %(message)s"
    )

# Default filenames
DEFAULT_CONFIG_FILENAME = "config.yml"
DEFAULT_COSTS_FILENAME = (
    "model_costs.json"  # Expected in 'data/' subdirectory of project root
)

# Environment variables for specifying config file paths explicitly
ENV_CONFIG_PATH = "LEAN_RANK_CONFIG_FILE"
ENV_COSTS_PATH = "LEAN_RANK_COSTS_FILE"

# Defines mappings from environment variables to nested configuration dictionary keys
# for OVERRIDING specific values within the loaded config.
# Format: (ENV_VARIABLE_NAME, [list, of, config, keys], target_type_for_conversion)
ENV_OVERRIDES: List[Tuple[str, List[str], Type]] = [
    ("LEAN_RANK_DB_URL", ["database", "url"], str),
    ("DEFAULT_GENERATION_MODEL", ["llm", "generation_model"], str),
    ("DEFAULT_EMBEDDING_MODEL", ["llm", "embedding_model"], str),
    ("GEMINI_MAX_RETRIES", ["llm", "retries"], int),
    ("GEMINI_BACKOFF_FACTOR", ["llm", "backoff"], float),
]


def _update_nested_dict(d: Dict[str, Any], keys: List[str], value: Any) -> None:
    """Safely sets a value in a nested dictionary based on a list of keys.

    Creates intermediate dictionaries if they don't exist. Logs an error
    if a path conflict occurs (e.g., expecting a dict but finding a non-dict).

    Args:
        d: The dictionary to update.
        keys: A list of strings representing the path to the key.
        value: The value to set at the specified path.
    """
    node = d
    for i, key in enumerate(keys[:-1]):
        child = node.setdefault(key, {})
        if not isinstance(child, dict):
            logger.error(
                "Config structure conflict: Expected dict at '%s' "
                "while setting path '%s', but found type %s. "
                "Cannot apply value '%s'.",
                key,
                ".".join(keys),
                type(child).__name__,
                value,
            )
            return
        node = child

    final_key = keys[-1]
    node[final_key] = value


def load_configuration(
    dotenv_path: Optional[str] = None,
    env_override_map: List[Tuple[str, List[str], Type]] = ENV_OVERRIDES,
) -> Dict[str, Any]:
    """Loads configuration layers: YAML, JSON costs, .env, and env overrides.

    Determines paths for 'config.yml' (project root) and
    'model_costs.json' (project_root/data/) using environment variables
    (LEAN_RANK_CONFIG_FILE, LEAN_RANK_COSTS_FILE), then by deriving the
    project root from this module's location (src/lean_explore/config.py),
    and finally falling back to the current working directory.

    Reads base config from YAML, merges 'costs' from JSON, loads .env file
    (if found or specified), and applies specific overrides from environment
    variables defined in `env_override_map`.

    Args:
        dotenv_path: Explicit path to a .env file. If None, `python-dotenv`
                     searches standard locations (e.g., project root).
        env_override_map: Mapping defining environment variable overrides for
                          configuration keys and their target types.

    Returns:
        Dict[str, Any]: The fully merged configuration. Returns an empty
        dictionary if the base config file cannot be loaded.
    """
    config: Dict[str, Any] = {}

    # --- Determine Configuration File Paths ---
    effective_config_path: Optional[pathlib.Path] = None
    effective_costs_path: Optional[pathlib.Path] = None

    # 1. Check Environment Variables for explicit paths
    env_config_path_str = os.getenv(ENV_CONFIG_PATH)
    env_costs_path_str = os.getenv(ENV_COSTS_PATH)

    if env_config_path_str:
        effective_config_path = pathlib.Path(env_config_path_str)
        logger.info(
            "Using config path from environment variable %s: '%s'",
            ENV_CONFIG_PATH,
            effective_config_path,
        )
    if env_costs_path_str:
        effective_costs_path = pathlib.Path(env_costs_path_str)
        logger.info(
            "Using costs path from environment variable %s: '%s'",
            ENV_COSTS_PATH,
            effective_costs_path,
        )

    # 2. Derive Project Root from script location (if paths not set by env vars)
    # Assumes this file is in src/lean_explore/config.py
    if effective_config_path is None or effective_costs_path is None:
        try:
            # Project root is parent.parent.parent of this file's location
            project_root = pathlib.Path(__file__).resolve().parent.parent.parent
            logger.debug("Determined project root: '%s'", project_root)

            if effective_config_path is None:
                candidate_path = project_root / DEFAULT_CONFIG_FILENAME
                if candidate_path.is_file():
                    effective_config_path = candidate_path
                    logger.info(
                        "Derived config path from project root: '%s'",
                        effective_config_path,
                    )
                else:
                    logger.debug(
                        "Config file not found at derived project path: '%s'",
                        candidate_path,
                    )

            if effective_costs_path is None:
                # Costs file is expected in 'data' subdirectory of project root
                candidate_path = project_root / "data" / DEFAULT_COSTS_FILENAME
                if candidate_path.is_file():
                    effective_costs_path = candidate_path
                    logger.info(
                        "Derived costs path from project root/data: '%s'",
                        effective_costs_path,
                    )
                else:
                    logger.debug(
                        "Costs file not found at derived project path: '%s'",
                        candidate_path,
                    )
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("Error determining project root or derived paths: %s", e)

    # 3. Fallback to Current Working Directory (if paths still not determined)
    if effective_config_path is None or effective_costs_path is None:
        logger.warning(
            "Could not determine config/costs path from env vars or project structure. "
            "Falling back to current working directory for any remaining unset paths."
        )
        cwd = pathlib.Path.cwd()
        if effective_config_path is None:
            effective_config_path = cwd / DEFAULT_CONFIG_FILENAME
            logger.info(
                "Using fallback config path in CWD: '%s'", effective_config_path
            )
        if effective_costs_path is None:
            # Costs file fallback to CWD/data/ (consistent with derived) or just CWD?
            # For simplicity, CWD directly for fallback if 'data' subfolder method is
            # too complex here.
            # Or, stick to project_root/data if project_root itself was CWD
            # The current logic already checks derived paths first. If project_root
            # was CWD, it would have checked CWD/data/...
            # So, a direct CWD fallback for costs might be CWD/model_costs.json
            # Let's keep costs fallback relative to CWD, in a 'data' subfolder if
            # that's intended structure,
            # or directly in CWD if the "data" subfolder is only for the
            # "project root" derived path.
            # For simplicity and consistency with config.yml, try
            # CWD/data/model_costs.json first, then CWD/model_costs.json
            costs_cwd_data_path = cwd / "data" / DEFAULT_COSTS_FILENAME
            costs_cwd_direct_path = cwd / DEFAULT_COSTS_FILENAME
            if costs_cwd_data_path.is_file():
                effective_costs_path = costs_cwd_data_path
                logger.info(
                    "Using fallback costs path in CWD/data: '%s'", effective_costs_path
                )
            elif costs_cwd_direct_path.is_file():
                effective_costs_path = costs_cwd_direct_path
                logger.info(
                    "Using fallback costs path in CWD: '%s'", effective_costs_path
                )
            else:
                logger.info(
                    "Fallback costs path in CWD (or CWD/data) not found either."
                )

    # --- Load Base YAML Configuration ---
    if effective_config_path:
        resolved_config_path = effective_config_path.resolve()
        try:
            with open(resolved_config_path, encoding="utf-8") as f:
                loaded_yaml = yaml.safe_load(f)
            config = loaded_yaml if isinstance(loaded_yaml, dict) else {}
            logger.info("Loaded base config from '%s'.", resolved_config_path)
        except FileNotFoundError:
            logger.warning("Base config file '%s' not found.", resolved_config_path)
            config = {}  # Initialize if base file is missing but path was determined
        except yaml.YAMLError as e:
            logger.error(
                "Error parsing YAML '%s': %s", resolved_config_path, e, exc_info=True
            )
            return {}  # Critical error, return empty
        except Exception as e:  # pylint: disable=broad-except
            logger.error(
                "Unexpected error loading '%s': %s",
                resolved_config_path,
                e,
                exc_info=True,
            )
            return {}
    else:
        logger.error(
            "Could not determine a valid path for the base configuration file. "
            "Cannot load configuration."
        )
        return {}  # Cannot proceed

    # --- Load and Merge JSON Costs Data ---
    if effective_costs_path:
        resolved_costs_path = effective_costs_path.resolve()
        try:
            with open(resolved_costs_path, encoding="utf-8") as f:
                model_costs = json.load(f)
            config.setdefault("costs", {}).update(
                model_costs if isinstance(model_costs, dict) else {}
            )
            logger.info("Loaded and merged costs from '%s'.", resolved_costs_path)
        except FileNotFoundError:
            logger.warning(
                "Costs file '%s' not found. 'costs' section may be incomplete.",
                resolved_costs_path,
            )
            config.setdefault("costs", {})
        except json.JSONDecodeError as e:
            logger.error(
                "Error parsing JSON costs '%s': %s",
                resolved_costs_path,
                e,
                exc_info=True,
            )
            config.setdefault("costs", {})
        except Exception as e:  # pylint: disable=broad-except
            logger.error(
                "Unexpected error loading '%s': %s",
                resolved_costs_path,
                e,
                exc_info=True,
            )
            config.setdefault("costs", {})
    else:
        logger.warning(
            "Could not determine a valid path for the costs file. 'costs' section "
            "may be empty or missing."
        )
        config.setdefault("costs", {})

    # --- Load .env file into environment variables ---
    try:
        # python-dotenv searches for .env in CWD or parent directories.
        # override=True means .env vars will take precedence over existing OS env vars.
        loaded_env = load_dotenv(dotenv_path=dotenv_path, verbose=False, override=True)
        if loaded_env:
            logger.info(
                ".env file loaded successfully (values override existing env vars)."
            )
        else:
            logger.debug(".env file not found at specified path or standard locations.")
    except Exception as e:  # pylint: disable=broad-except
        logger.error("Error loading .env file: %s", e, exc_info=True)

    # --- Apply Environment Variable Overrides for config VALUES ---
    logger.debug("Checking for environment variable overrides for config values...")
    override_count = 0
    for env_var, config_keys, target_type in env_override_map:
        env_value_str = os.getenv(env_var)
        if env_value_str is not None:
            try:
                typed_value = target_type(env_value_str)
                _update_nested_dict(config, config_keys, typed_value)
                logger.info(
                    "Applied config override: '%s' = '%s' (from env '%s')",
                    ".".join(config_keys),
                    typed_value,
                    env_var,
                )
                override_count += 1
            except ValueError:
                logger.warning(
                    "Value override failed: Cannot convert env var '%s' value '%s' "
                    "to type %s.",
                    env_var,
                    env_value_str,
                    target_type.__name__,
                )
            except Exception as e:  # pylint: disable=broad-except
                logger.error(
                    "Value override error: Unexpected issue applying env var '%s': %s",
                    env_var,
                    e,
                    exc_info=True,
                )
    if override_count > 0:
        logger.info(
            "Applied %d environment variable value override(s).", override_count
        )
    else:
        logger.debug("No environment variable value overrides applied from map.")

    return config


# --- Singleton Configuration Instance ---
# Load the configuration once when this module is first imported.
APP_CONFIG: Dict[str, Any] = load_configuration()


# --- Direct Accessors for Sensitive/Specific Environment Variables ---


def get_gemini_api_key() -> Optional[str]:
    """Retrieves the Gemini API Key directly from environment variables.

    This function bypasses the main `APP_CONFIG` for direct access to the
    `GEMINI_API_KEY` environment variable, typically loaded via `.env` or
    set in the execution environment.

    Returns:
        Optional[str]: The Gemini API key string if the 'GEMINI_API_KEY'
        environment variable is set, otherwise None.
    """
    return os.getenv("GEMINI_API_KEY")


# --- Example Usage / Standalone Test ---
if __name__ == "__main__":
    # This block executes only when the script is run directly for testing.
    # It re-configures logging for standalone clarity.
    logging.basicConfig(
        level=logging.DEBUG, format="%(levelname)-8s [%(name)s:%(lineno)d] %(message)s"
    )
    logger.info("\n--- Running Config Loader Standalone Test ---")

    # Demonstrate path determination logic (simplified for test)
    logger.info("\n--- Path Determination (Test Simulation) ---")
    test_cfg_path: Optional[pathlib.Path] = None
    test_csts_path: Optional[pathlib.Path] = None
    env_cfg_var = os.getenv(ENV_CONFIG_PATH)
    env_csts_var = os.getenv(ENV_COSTS_PATH)

    if env_cfg_var:
        test_cfg_path = pathlib.Path(env_cfg_var)
    if env_csts_var:
        test_csts_path = pathlib.Path(env_csts_var)

    if not test_cfg_path or not test_csts_path:
        try:
            # This path needs to match the one in load_configuration
            current_module_path = pathlib.Path(__file__).resolve()
            # src/lean_explore/config.py -> src/lean_explore -> src -> project_root
            _test_project_root = current_module_path.parent.parent.parent
            if not test_cfg_path:
                _p = _test_project_root / DEFAULT_CONFIG_FILENAME
                if _p.is_file():
                    test_cfg_path = _p
            if not test_csts_path:
                _p = _test_project_root / "data" / DEFAULT_COSTS_FILENAME
                if _p.is_file():
                    test_csts_path = _p
        except Exception:  # pylint: disable=broad-except
            logger.debug(
                "Error finding project root in standalone test.", exc_info=True
            )

    if not test_cfg_path or not test_csts_path:
        _cwd = pathlib.Path.cwd()
        if not test_cfg_path:
            test_cfg_path = _cwd / DEFAULT_CONFIG_FILENAME
        if not test_csts_path:
            _p_data = _cwd / "data" / DEFAULT_COSTS_FILENAME
            _p_direct = _cwd / DEFAULT_COSTS_FILENAME
            if _p_data.is_file():
                test_csts_path = _p_data
            elif _p_direct.is_file():
                test_csts_path = _p_direct

    logger.info(
        "Config Path Effective: %s",
        test_cfg_path.resolve()
        if test_cfg_path and test_cfg_path.exists()
        else "Not Found or Determined",
    )
    logger.info(
        "Costs Path Effective: %s",
        test_csts_path.resolve()
        if test_csts_path and test_csts_path.exists()
        else "Not Found or Determined",
    )

    # APP_CONFIG is already loaded when the module is imported.
    logger.info("\n--- Loaded Configuration (APP_CONFIG from module import) ---")
    import pprint  # pylint: disable=import-outside-toplevel

    pprint.pprint(APP_CONFIG)

    logger.info("\n--- Accessing Values Example ---")
    db_url = APP_CONFIG.get("database", {}).get("url", "N/A")
    gen_model = APP_CONFIG.get("llm", {}).get("generation_model", "N/A")
    logger.info("Database URL: %s", db_url)
    logger.info("Generation Model: %s", gen_model)

    cost_info = APP_CONFIG.get("costs", {}).get("gemini-1.5-flash-latest", {})
    logger.info("Costs for gemini-1.5-flash-latest: %s", cost_info or "N/A")

    logger.info("\n--- Accessing Secrets Example ---")
    api_key_val = get_gemini_api_key()
    logger.info("Gemini API Key is set: %s", bool(api_key_val))

    logger.info("\n--- Standalone Test Complete ---")

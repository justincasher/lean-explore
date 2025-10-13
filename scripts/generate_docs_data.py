# scripts/generate_docs_data.py

"""Generates structured documentation data from Python source files.

This script utilizes the 'griffe' library to parse a specified Python package.
It extracts comprehensive information about all modules, classes, functions,
and their docstrings, assuming compatibility with Google style. This information
is then serialized into a JSON file. The JSON file can be consumed by a
frontend application, such as a Vue.js website, to display interactive API
documentation.

The script recursively traverses the package structure to ensure all nested
modules are processed and included in the output.
"""

import json
import logging
import pathlib
from typing import Any, Dict, List, Optional, Union, cast

from griffe import (
    Alias,
    AliasResolutionError,
    Class,
    Decorator,
    DocstringNamedElement,
    DocstringSectionAdmonition,
    DocstringSectionAttributes,
    DocstringSectionClasses,
    DocstringSectionDeprecated,
    DocstringSectionExamples,
    DocstringSectionFunctions,
    DocstringSectionModules,
    DocstringSectionOtherParameters,
    DocstringSectionParameters,
    DocstringSectionRaises,
    DocstringSectionReceives,
    DocstringSectionReturns,
    DocstringSectionText,
    DocstringSectionWarns,
    DocstringSectionYields,
    Expr,
    ExprCall,
    Function,
    GriffeLoader,
    Module,
    Parameters,
    get_logger,
)

# Configure logging for griffe and this script.
get_logger().setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# --- Configuration ---
PACKAGE_PATH = pathlib.Path("src/lean_explore")
"""Path to the root of the Python package to document."""

OUTPUT_PATH = pathlib.Path("data/module_data.json")
"""Output file path for the generated JSON data."""
# --- End Configuration ---


def _resolve_annotation(annotation: Union[str, Expr, None]) -> str:
    """Resolves a griffe annotation object to its string representation.

    Args:
        annotation (Union[str, Expr, None]): The annotation, which can be a
            string, a griffe Expr object, or None.

    Returns:
        str: The string representation of the annotation, or an empty string if
        the annotation is None.
    """
    if isinstance(annotation, Expr):
        return str(annotation)
    if isinstance(annotation, str):
        return annotation
    return ""


def _parse_docstring_sections(
    docstring_obj: Optional[Any],
) -> Dict[str, Any]:
    """Parses standard sections from a griffe Docstring object.

    This function processes common docstring sections like summary, parameters,
    returns, attributes, raises, examples, and various admonitions (notes,
    warnings, etc.), transforming them into a structured dictionary.

    Args:
        docstring_obj (Optional[Any]): A griffe Docstring object, which
            contains a 'parsed' attribute with a list of DocstringSection
            instances, or None.

    Returns:
        Dict[str, Any]: A dictionary where keys are section kinds (e.g.,
        'summary', 'parameters', 'raises', 'note') and values are the
        structured content of these sections.
    """
    if not docstring_obj or not hasattr(docstring_obj, "parsed"):
        return {}

    sections_data: Dict[str, Any] = {}
    summary = ""
    full_text_parts = []

    for section in docstring_obj.parsed:
        section_kind_str = section.kind.value  # e.g., "text", "parameters", "returns"

        if isinstance(section, DocstringSectionText):
            if not summary:
                summary = section.value.strip()
            full_text_parts.append(section.value.strip())
        elif isinstance(section, DocstringSectionParameters):
            params = [
                {
                    "name": item.name,
                    "annotation": _resolve_annotation(item.annotation),
                    "description": item.description.strip() if item.description else "",
                    "value": str(item.value) if item.value is not None else None,
                }
                for item in section.value  # item is DocstringParameter
            ]
            sections_data[section_kind_str] = params
        elif isinstance(section, DocstringSectionReturns):
            returns_data = [
                {
                    "name": item.name
                    if hasattr(item, "name")
                    else "",  # DocstringReturn items may not have a name
                    "annotation": _resolve_annotation(item.annotation),
                    "description": item.description.strip() if item.description else "",
                }
                for item in section.value  # item is DocstringReturn
            ]
            # If only one return entry, store as object, else as list.
            if len(returns_data) == 1:
                sections_data[section_kind_str] = returns_data[0]
            elif len(returns_data) > 1:
                sections_data[section_kind_str] = returns_data
        elif isinstance(section, DocstringSectionAttributes):
            attrs = [
                {
                    "name": item.name,
                    "annotation": _resolve_annotation(item.annotation),
                    "description": item.description.strip() if item.description else "",
                }
                for item in section.value  # item is DocstringAttribute
            ]
            sections_data[section_kind_str] = attrs
        elif isinstance(
            section, DocstringSectionRaises
        ):  # Specific handling for "Raises"
            raises_data = []
            for item in section.value:  # item is DocstringRaise
                # DocstringRaise has 'annotation' (for exception type) and
                # 'description'
                if hasattr(item, "annotation") and hasattr(item, "description"):
                    raises_data.append(
                        {
                            "type": _resolve_annotation(item.annotation),
                            "description": item.description.strip()
                            if item.description
                            else "",
                        }
                    )
                else:  # Fallback if item is not structured as an expected
                    # DocstringRaise
                    logging.warning(
                        "Unexpected item structure in DocstringSectionRaises: "
                        f"{str(item)[:100]}"
                    )
                    raises_data.append(
                        {"type": "UnknownException", "description": str(item)}
                    )
            sections_data[section_kind_str] = raises_data
        elif isinstance(section, DocstringSectionExamples):
            examples = [
                {
                    "title": item.title.strip() if item.title else None,
                    "code": item.value.strip(),
                }
                for item in section.value
            ]
            sections_data[section_kind_str] = examples
        elif isinstance(section, DocstringSectionAdmonition):
            admonition_payload = section.value

            if (
                hasattr(admonition_payload, "kind")
                and isinstance(admonition_payload.kind, str)
                and hasattr(admonition_payload, "text")
                and isinstance(admonition_payload.text, str)
            ):
                parsed_admonition_kind_str = admonition_payload.kind
                admonition_text_content = admonition_payload.text

                admonition_data_item = {
                    "title": section.title.strip()
                    if section.title
                    else parsed_admonition_kind_str,
                    "text": admonition_text_content.strip(),
                }

                if parsed_admonition_kind_str not in sections_data:
                    sections_data[parsed_admonition_kind_str] = []
                sections_data[parsed_admonition_kind_str].append(admonition_data_item)
            else:
                logging.warning(
                    "Unexpected payload structure for DocstringSectionAdmonition "
                    f"(title: '{section.title}', section kind from parser: "
                    f"'{section_kind_str}'). Payload type: "
                    f"{type(admonition_payload)}. Skipping this admonition section."
                )
        elif isinstance(
            section,
            (
                DocstringSectionDeprecated,
                DocstringSectionWarns,
                DocstringSectionYields,
                DocstringSectionReceives,  # DocstringSectionRaises handled above
                DocstringSectionOtherParameters,
                DocstringSectionClasses,
                DocstringSectionFunctions,
                DocstringSectionModules,
            ),
        ):
            # Generic handling for other structured list-based sections
            if hasattr(section, "value") and isinstance(section.value, list):
                values = []
                for item in section.value:
                    if isinstance(
                        item, DocstringNamedElement
                    ):  # For sections listing named elements
                        values.append(
                            {
                                "name": item.name,
                                "annotation": _resolve_annotation(item.annotation)
                                if hasattr(item, "annotation")
                                else "",
                                "description": item.description.strip()
                                if item.description
                                else "",
                            }
                        )
                    # DocstringWarn, DocstringYield, DocstringReceive
                    # might have name/description
                    elif hasattr(item, "name") and hasattr(item, "description"):
                        values.append(
                            {
                                "name": item.name,
                                "description": item.description.strip()
                                if item.description
                                else "",
                            }
                        )
                    elif hasattr(
                        item, "text"
                    ):  # For simpler text items in a list section
                        values.append(item.text.strip())
                    else:  # Fallback if item structure is unknown within the list
                        values.append(str(item))
                sections_data[section_kind_str] = values
            elif hasattr(section, "value"):  # For sections with a single non-list value
                sections_data[section_kind_str] = str(section.value)
        else:  # Fallback for any completely unrecognized section types
            if hasattr(section, "value"):
                sections_data[section_kind_str] = str(section.value)
            else:
                sections_data[section_kind_str] = "Unsupported section structure"

    if summary:
        sections_data["summary"] = summary

    # Consolidate all text parts into a main "text" field.
    # If summary is already the start of combined text, don't prepend.
    text_content = "\n\n".join(part for part in full_text_parts if part)
    if summary and text_content.strip().startswith(
        summary.strip()
    ):  # More robust check
        sections_data["text"] = text_content
    elif summary:  # Prepend summary if it's distinct or text_content is empty
        sections_data["text"] = (
            f"{summary}\n\n{text_content}".strip() if text_content else summary
        )
    else:
        sections_data["text"] = text_content

    return sections_data


# Note: The original code had two definitions of _serialize_function.
# The second one would effectively overwrite the first in Python.
# I am providing the docstrings updated for both, as per the instruction
# not to change the code structure.


def _serialize_function(func_obj: Function, module_path: str) -> Dict[str, Any]:
    """Serializes a griffe Function object, representing a function or method.

    This function combines information extracted from the code (such as the
    signature) with details parsed from its docstring. For the 'returns'
    section, if multiple return descriptions are found in the docstring,
    they are concatenated.

    Args:
        func_obj (Function): The griffe Function object to serialize.
        module_path (str): The canonical path of the parent module or class.
            This is used for context, though `func_obj.canonical_path`
            is generally preferred for the function's own path.

    Returns:
        Dict[str, Any]: A dictionary representing the function or method.
        This includes its name, canonical path, parsed docstring sections,
        parameters (from code, enhanced with docstring descriptions), return
        information (annotation from code, description from docstring),
        decorators, asynchronous status, and source file location.
    """
    docstring_sections = _parse_docstring_sections(func_obj.docstring)
    code_signature_params = _serialize_parameters(func_obj.parameters)

    docstring_params_map = {
        p["name"]: p for p in docstring_sections.get("parameters", [])
    }
    for param_info in code_signature_params:
        if param_info["name"] in docstring_params_map:
            param_info["description"] = docstring_params_map[param_info["name"]].get(
                "description", ""
            )

    # Initialize returns_info with annotation from code signature
    returns_info = {
        "annotation": _resolve_annotation(func_obj.returns),
        "description": "",  # To be populated from docstring
    }

    # Populate description from parsed docstring "returns" section
    docstring_returns_section = docstring_sections.get("returns")
    if isinstance(docstring_returns_section, dict):  # Single return item in docstring
        returns_info["description"] = docstring_returns_section.get(
            "description", ""
        ).strip()
        # The main 'annotation' in returns_info prioritizes the code signature.
        # The docstring_sections["returns"] (if it's a dict) will also have an
        # 'annotation' field parsed from the docstring, which can be used by
        # the frontend if needed.
    elif isinstance(docstring_returns_section, list) and docstring_returns_section:
        # Concatenate all descriptions from the list of return items in the
        # docstring
        all_descriptions = [
            item.get("description", "").strip()
            for item in docstring_returns_section
            if item.get(
                "description", ""
            ).strip()  # Ensure item and description exist and are not empty
        ]
        if all_descriptions:
            returns_info["description"] = " ".join(all_descriptions)
        # Similar to the single item case, the main 'annotation' prioritizes
        # code signature. The list in docstring_sections["returns"] contains
        # individual annotations and descriptions.

    return {
        "name": func_obj.name,
        "path": func_obj.canonical_path,
        "docstring": func_obj.docstring.value if func_obj.docstring else "",
        "docstring_sections": docstring_sections,
        "parameters": code_signature_params,
        "returns": returns_info,
        "decorators": _serialize_decorators(func_obj.decorators),
        "is_async": getattr(func_obj, "is_async", False),
        "filepath": str(func_obj.filepath.relative_to(pathlib.Path.cwd()))
        if func_obj.filepath
        else None,
        "lineno": func_obj.lineno,
        "lines": [func_obj.lineno, func_obj.endlineno]
        if func_obj.lineno and func_obj.endlineno
        else [],
    }


def _serialize_parameters(parameters: Parameters) -> List[Dict[str, Any]]:
    """Serializes a griffe Parameters object from a code signature.

    Args:
        parameters (Parameters): A griffe Parameters object representing the
            parameters of a function or method as extracted from the source
            code.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, where each dictionary
        represents a parameter. Each parameter dictionary includes its name,
        type annotation string, kind (e.g., positional, keyword), and
        default value string. For 'typer.Option' calls, it attempts to
        format the default value string as multi-line.
    """
    if not parameters:
        return []

    serialized_params: List[Dict[str, Any]] = []
    for param in parameters:
        default_value_repr: Optional[str] = None
        if param.default is not None:
            if (
                isinstance(param.default, ExprCall)
                and hasattr(param.default, "function")
                and str(param.default.function) == "typer.Option"
            ):
                func_name_str = str(param.default.function)

                args_str_list: List[str] = []
                if hasattr(param.default, "arguments"):
                    for arg_expr in param.default.arguments:
                        args_str_list.append(str(arg_expr))

                if not args_str_list:
                    default_value_repr = f"{func_name_str}()"
                else:
                    arg_indent = "    "
                    formatted_args = f",\n{arg_indent}".join(args_str_list)
                    default_value_repr = (
                        f"{func_name_str}(\n{arg_indent}{formatted_args}\n)"
                    )

            else:
                default_value_repr = str(param.default)

        serialized_params.append(
            {
                "name": param.name,
                "annotation": _resolve_annotation(param.annotation),
                "kind": param.kind.value,
                "default": default_value_repr,
            }
        )
    return serialized_params


def _serialize_decorators(decorators: List[Decorator]) -> List[Dict[str, Any]]:
    """Serializes a list of griffe Decorator objects.

    Includes detailed error logging for cases where decorator information
    might be partially irretrievable.

    Args:
        decorators (List[Decorator]): A list of griffe Decorator objects.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each representing a
        decorator. Includes textual representation, callable path, and line
        numbers. Returns an empty list if the input list is empty.
    """
    if not decorators:
        return []

    serialized_decorators = []
    for i, dec in enumerate(decorators):
        decorator_textual_representation = "ERROR_RETRIEVING_TEXT"
        decorator_callable_path_str = "ERROR_RETRIEVING_PATH"
        dec_lineno = getattr(dec, "lineno", None)
        dec_endlineno = getattr(dec, "endlineno", None)

        try:
            decorator_textual_representation = str(dec.value)
        except AttributeError as e_val_attr:
            logging.error(
                "DECORATOR_SERIALIZATION_DEBUG: AttributeError while processing "
                f"str(dec.value) for decorator #{i}. Error: {e_val_attr}"
            )
        except Exception as e_val_other:
            logging.error(
                "DECORATOR_SERIALIZATION_DEBUG: Other exception while processing "
                f"str(dec.value) for decorator #{i}. Error: {e_val_other}"
            )

        try:
            decorator_callable_path_str = dec.callable_path
        except AttributeError as e_path_attr:
            logging.error(
                "DECORATOR_SERIALIZATION_DEBUG: AttributeError while processing "
                f"dec.callable_path for decorator #{i} (text was: "
                f"'{decorator_textual_representation}'). Error: {e_path_attr}"
            )
            if "'Decorator' object has no attribute 'name'" in str(e_path_attr):
                logging.error(
                    "--> Confirmed: dec.callable_path triggered the 'Decorator... "
                    "no attribute name' error."
                )
        except Exception as e_path_other:
            logging.error(
                "DECORATOR_SERIALIZATION_DEBUG: Other exception while processing "
                f"dec.callable_path for decorator #{i} (text was: "
                f"'{decorator_textual_representation}'). Error: {e_path_other}"
            )

        serialized_decorators.append(
            {
                "text": decorator_textual_representation,
                "path": decorator_callable_path_str,
                "lineno": dec_lineno,
                "endlineno": dec_endlineno,
            }
        )
    return serialized_decorators


def _serialize_function(func_obj: Function, module_path: str) -> Dict[str, Any]:
    """Serializes a griffe Function object, representing a function or method.

    This function combines information extracted from the code (such as the
    signature) with details parsed from its docstring.

    Args:
        func_obj (Function): The griffe Function object to serialize.
        module_path (str): The canonical path of the parent module or class.
            This is used for context, though `func_obj.canonical_path`
            is generally preferred for the function's own path.

    Returns:
        Dict[str, Any]: A dictionary representing the function or method.
        This includes its name, canonical path, parsed docstring sections,
        parameters (from code, enhanced with docstring descriptions), return
        information (from code and docstring), decorators, asynchronous
        status, and source file location.
    """
    docstring_sections = _parse_docstring_sections(func_obj.docstring)
    code_signature_params = _serialize_parameters(func_obj.parameters)

    docstring_params_map = {
        p["name"]: p for p in docstring_sections.get("parameters", [])
    }
    for param_info in code_signature_params:
        if param_info["name"] in docstring_params_map:
            param_info["description"] = docstring_params_map[param_info["name"]].get(
                "description", ""
            )

    returns_info = {
        "annotation": _resolve_annotation(func_obj.returns),
        "description": "",
    }
    docstring_returns_section = docstring_sections.get("returns")
    if isinstance(docstring_returns_section, dict):
        returns_info["description"] = docstring_returns_section.get("description", "")
        if docstring_returns_section.get("annotation"):
            returns_info["annotation"] = docstring_returns_section.get("annotation")
    elif isinstance(docstring_returns_section, list) and docstring_returns_section:
        first_return_item = docstring_returns_section[0]
        returns_info["description"] = first_return_item.get("description", "")
        if first_return_item.get("annotation"):
            returns_info["annotation"] = first_return_item.get("annotation")
        if len(docstring_returns_section) > 1:
            returns_info["description"] += " (Multiple return paths documented)"

    return {
        "name": func_obj.name,
        "path": func_obj.canonical_path,
        "docstring": func_obj.docstring.value if func_obj.docstring else "",
        "docstring_sections": docstring_sections,
        "parameters": code_signature_params,
        "returns": returns_info,
        "decorators": _serialize_decorators(func_obj.decorators),
        "is_async": getattr(func_obj, "is_async", False),
        "filepath": str(func_obj.filepath.relative_to(pathlib.Path.cwd()))
        if func_obj.filepath
        else None,
        "lineno": func_obj.lineno,
        "lines": [func_obj.lineno, func_obj.endlineno]
        if func_obj.lineno and func_obj.endlineno
        else [],
    }


def _serialize_module(module_obj: Module) -> Dict[str, Any]:
    """Serializes the immediate contents of a griffe Module object.

    This function processes the direct members of the given module, such as
    its top-level functions and classes that are *defined* within this module.
    It extracts their documentation details and structures them into a dictionary.
    Imported items (aliases) are not included in the lists of functions/classes
    for this specific module, as they are defined elsewhere.

    Note:
        Traversal of submodules and their serialization into the main list
        are handled by the calling logic in the `main` function.

    Args:
        module_obj (Module): The griffe Module object to serialize.

    Returns:
        Dict[str, Any]: A dictionary representing the module, including its
        name, canonical path, filepath, parsed docstring, lists of
        serialized functions and classes (defined in this module), and line number.
    """
    functions_data = []
    classes_data = []
    current_module_canonical_path = module_obj.canonical_path

    for member_name, member_obj_inner in module_obj.members.items():
        try:
            target_object: Optional[Union[Function, Class, Module, Alias]] = None
            if member_obj_inner.is_alias:
                # Resolve the alias to its final target
                target_object = member_obj_inner.final_target
            else:
                target_object = member_obj_inner

            if not target_object:
                continue

            # Determine the canonical path of the module where the target object is
            # defined
            if not isinstance(target_object.canonical_path, str):
                logging.debug(
                    f"Skipping member '{member_name}' in module "
                    f"'{current_module_canonical_path}' because its target's "
                    f"canonical path is not a string: {target_object.canonical_path}"
                )
                continue

            # Use the parent's canonical path for a more direct check of definition
            # scope
            defined_in_module_path = ""
            if (
                hasattr(target_object, "parent")
                and target_object.parent
                and hasattr(target_object.parent, "canonical_path")
            ):
                defined_in_module_path = target_object.parent.canonical_path
            elif (  # Fallback for items that might not have parent set as expected
                "." in target_object.canonical_path
            ):
                defined_in_module_path = target_object.canonical_path.rsplit(".", 1)[0]
            else:  # Likely a top-level module itself, not a function/class defined
                # in another module
                defined_in_module_path = target_object.canonical_path

            if target_object.is_function and isinstance(target_object, Function):
                # Only include if defined in the current module
                if defined_in_module_path == current_module_canonical_path:
                    functions_data.append(
                        _serialize_function(
                            target_object, current_module_canonical_path
                        )
                    )
                else:
                    logging.debug(
                        f"Function '{target_object.name}' (from "
                        f"{defined_in_module_path}) is imported into "
                        f"'{current_module_canonical_path}' and will not be listed "
                        "directly under it."
                    )
            elif target_object.is_class and isinstance(target_object, Class):
                # Only include if defined in the current module
                if defined_in_module_path == current_module_canonical_path:
                    classes_data.append(
                        _serialize_class(target_object, current_module_canonical_path)
                    )
                else:
                    logging.debug(
                        f"Class '{target_object.name}' (from "
                        f"{defined_in_module_path}) is imported into "
                        f"'{current_module_canonical_path}' and will not be listed "
                        "directly under it."
                    )

        except AliasResolutionError:
            a = member_obj_inner.target_path if member_obj_inner.is_alias else "self"
            logging.warning(
                f"Skipping member '{member_obj_inner.name}' in module "
                f"'{current_module_canonical_path}' due to AliasResolutionError. "
                f"Target: '{a}'."
            )
            continue
        except AttributeError as e_attr:
            logging.error(
                f"Caught AttributeError for member '{member_obj_inner.name}' in module "
                f"'{current_module_canonical_path}'. "
                f"Message: '{e_attr}'. Skipping this member."
            )
            continue

    return {
        "name": module_obj.name,
        "path": module_obj.canonical_path,
        "filepath": str(module_obj.filepath.relative_to(pathlib.Path.cwd()))
        if module_obj.filepath
        else None,
        "docstring": module_obj.docstring.value if module_obj.docstring else "",
        "docstring_sections": _parse_docstring_sections(module_obj.docstring),
        "functions": sorted(functions_data, key=lambda x: x["name"]),
        "classes": sorted(classes_data, key=lambda x: x["name"]),
        "lineno": module_obj.lineno,
    }


def _serialize_class(class_obj: Class, module_path: str) -> Dict[str, Any]:
    """Serializes a griffe Class object.

    This function processes the members of the given class, such as its
    methods and attributes (both from code and docstrings). Information
    from code analysis and docstrings is combined. Aliases for methods
    that cannot be resolved are logged and skipped.

    Args:
        class_obj (Class): The griffe Class object to serialize.
        module_path (str): The canonical path of the module containing this
            class. This is used for context, though `class_obj.canonical_path`
            is primary for the class itself.

    Returns:
        Dict[str, Any]: A dictionary representing the class, including its name,
        canonical path, parsed docstring, lists of serialized methods and
        attributes, base classes, source file location, and line numbers.
    """
    methods_data = []
    attributes_data = []

    for member_name, member_obj in class_obj.members.items():
        if member_obj.is_attribute:
            attributes_data.append(
                {
                    "name": member_obj.name,
                    "value": str(member_obj.value)
                    if member_obj.value is not None
                    else None,
                    "annotation": _resolve_annotation(member_obj.annotation),
                    "docstring": member_obj.docstring.value
                    if member_obj.docstring
                    else "",
                    "path": member_obj.canonical_path,
                    "filepath": str(member_obj.filepath.relative_to(pathlib.Path.cwd()))
                    if member_obj.filepath
                    else None,
                    "lineno": member_obj.lineno,
                }
            )
        elif member_obj.is_function:
            try:
                actual_method_to_serialize: Optional[Function] = None
                if member_obj.is_alias:
                    actual_method_to_serialize = cast(Function, member_obj.final_target)
                else:
                    actual_method_to_serialize = cast(Function, member_obj)

                if actual_method_to_serialize:
                    methods_data.append(
                        _serialize_function(
                            actual_method_to_serialize, class_obj.canonical_path
                        )
                    )
            except AliasResolutionError:
                a = member_obj.target_path
                b = member_obj.is_alias
                logging.warning(
                    f"Skipping method (member) '{member_obj.name}' in class "
                    f"'{class_obj.canonical_path}' due to AliasResolutionError. "
                    "It likely points to an external or unresolvable callable "
                    f"(target: '{a if b else 'self'}')."
                )

    docstring_sections = _parse_docstring_sections(class_obj.docstring)

    # Enhance attributes_data with attributes found only in docstrings,
    # or add docstrings to attributes found in code.
    docstring_attrs_list = docstring_sections.get("attributes", [])
    existing_code_attr_names = {attr["name"] for attr in attributes_data}
    for ds_attr_info in docstring_attrs_list:
        if ds_attr_info["name"] not in existing_code_attr_names:
            attributes_data.append(
                {
                    "name": ds_attr_info["name"],
                    "value": None,
                    "annotation": ds_attr_info.get("annotation", ""),
                    "docstring": ds_attr_info.get("description", ""),
                    "path": f"{class_obj.canonical_path}.{ds_attr_info['name']}",
                    "filepath": None,
                    "lineno": None,
                }
            )
        else:
            # If attribute was found in code, try to add its docstring description.
            for code_attr in attributes_data:
                if (
                    code_attr["name"] == ds_attr_info["name"]
                    and not code_attr["docstring"]
                ):
                    code_attr["docstring"] = ds_attr_info.get("description", "")
                    break

    return {
        "name": class_obj.name,
        "path": class_obj.canonical_path,
        "docstring": class_obj.docstring.value if class_obj.docstring else "",
        "docstring_sections": docstring_sections,
        "methods": sorted(
            methods_data, key=lambda x: (x["name"] != "__init__", x["name"])
        ),
        "attributes": sorted(attributes_data, key=lambda x: x["name"]),
        "bases": [_resolve_annotation(base) for base in class_obj.bases],
        "filepath": str(class_obj.filepath.relative_to(pathlib.Path.cwd()))
        if class_obj.filepath
        else None,
        "lineno": class_obj.lineno,
        "lines": [class_obj.lineno, class_obj.endlineno]
        if class_obj.lineno and class_obj.endlineno
        else [],
    }


def main() -> None:
    """Loads Python package structure and docstrings, then serializes to JSON.

    The script initializes a GriffeLoader to parse the Python package
    defined by `PACKAGE_PATH`. It then recursively traverses the package
    structure. For each module belonging to the target package, it
    serializes its documentation details (docstring, functions, classes, etc.)
    using helper functions.

    The collected data for all relevant modules is compiled into a list,
    sorted by module path, and then written to the JSON file specified
    by `OUTPUT_PATH`.
    """
    logging.info(f"Starting documentation generation for package: {PACKAGE_PATH}")
    loader = GriffeLoader(
        search_paths=[str(PACKAGE_PATH.parent)], docstring_parser="google"
    )

    package_name = ""
    root_package_obj: Optional[Union[Module, Alias]] = None
    try:
        package_name = PACKAGE_PATH.name
        root_package_obj = loader.load(package_name)
        if root_package_obj:
            # Resolve aliases within the loaded package, excluding external ones.
            loader.resolve_aliases(implicit=True, external=False)
        else:
            logging.error(
                f"Failed to load root package object for '{package_name}'."
                "Griffe loader.load() returned None."
            )
            return
    except Exception as e:
        logging.error(
            f"Failed to load package '{package_name or str(PACKAGE_PATH)}' "
            f"with Griffe: {e}"
        )
        logging.exception("Griffe loading error details:")
        return

    all_modules_data: List[Dict[str, Any]] = []
    processed_paths: set[str] = set()

    def collect_all_modules_recursively(
        current_module_obj: Module,
    ) -> None:
        """Recursively collects and serializes module data.

        This function checks if a module has already been processed. If not,
        it verifies if the module belongs to the target package (defined by
        `PACKAGE_PATH`). If it does, the module is serialized and added to
        `all_modules_data`. The function then iterates through the module's
        members. If any member is a submodule (or an alias to a module
        that can be resolved within the package context), it makes a
        recursive call to process that submodule. Unresolvable aliases,
        especially to external modules, are skipped.

        Args:
            current_module_obj (Module): The griffe Module object to process.
        """
        if (
            not current_module_obj
            or not hasattr(current_module_obj, "canonical_path")
            or current_module_obj.canonical_path in processed_paths
        ):
            return

        module_canonical_name = current_module_obj.canonical_path

        # Determine if the current module is part of the target package.
        is_target_module = False
        if current_module_obj.filepath:
            try:
                # Compare resolved absolute filepaths.
                abs_module_filepath = current_module_obj.filepath.resolve(strict=True)
                abs_package_path = PACKAGE_PATH.resolve(strict=True)
                if (
                    abs_package_path == abs_module_filepath
                    or abs_package_path in abs_module_filepath.parents
                ):
                    is_target_module = True
            except FileNotFoundError:
                logging.debug(
                    "File not found for module path "
                    f"{current_module_obj.filepath}, cannot confirm it's a "
                    "target module by path during recursion."
                )
            except Exception as path_exc:  # Catch other path resolution errors.
                logging.debug(
                    f"Could not resolve/compare paths for {current_module_obj.filepath}"
                    f" during recursion: {path_exc}"
                )

        # Fallback check based on canonical name if filepath check was inconclusive.
        if (
            not is_target_module
            and package_name
            and isinstance(module_canonical_name, str)
            and module_canonical_name.startswith(package_name)
        ):
            is_target_module = True

        if not is_target_module:
            logging.debug(
                "Skipping recursive processing for module (not in target package): "
                f"{module_canonical_name}"
            )
            return

        logging.info(
            f"Processing module: {module_canonical_name} "
            f"(File: {current_module_obj.filepath})"
        )
        processed_paths.add(module_canonical_name)

        serialized_module = _serialize_module(current_module_obj)
        all_modules_data.append(serialized_module)

        # Recursively process member submodules.
        for member_obj in current_module_obj.members.values():
            actual_member_obj_to_recurse: Optional[Module] = None
            try:
                if member_obj.is_module:
                    if member_obj.is_alias:
                        actual_member_obj_to_recurse = cast(
                            Module, member_obj.final_target
                        )
                    else:
                        actual_member_obj_to_recurse = cast(Module, member_obj)

            except AliasResolutionError:
                a = member_obj.target_path
                b = member_obj.is_alias
                logging.warning(
                    f"Skipping recursive check for member "
                    f"'{member_obj.name}' in module "
                    f"'{current_module_obj.canonical_path}' due to "
                    "AliasResolutionError "
                    "when determining if it's a module "
                    f"(target: '{a if b else 'self'}')."
                )
            except AttributeError as e_attr:
                logging.error(
                    f"Caught AttributeError for member '{member_obj.name}' in module "
                    f"'{current_module_obj.canonical_path}' "
                    "during module check. Message: "
                    f"'{e_attr}'"
                )
            except Exception as e_broad:
                logging.error(
                    "Caught a general exception of type "
                    f"'{type(e_broad).__name__}' for member '{member_obj.name}' "
                    f"in module '{current_module_obj.canonical_path}' during module "
                    f"check. Message: '{e_broad}'"
                )

            if actual_member_obj_to_recurse:
                collect_all_modules_recursively(actual_member_obj_to_recurse)

    actual_root_module_to_process: Optional[Module] = None
    if root_package_obj:
        if isinstance(root_package_obj, Module):
            actual_root_module_to_process = root_package_obj
        elif (
            isinstance(root_package_obj, Alias)
            and hasattr(root_package_obj, "resolved_target")
            and isinstance(root_package_obj.resolved_target, Module)
        ):
            actual_root_module_to_process = cast(
                Module, root_package_obj.resolved_target
            )

    if actual_root_module_to_process:
        logging.info(
            "Starting recursive module collection from root: "
            f"{actual_root_module_to_process.canonical_path}"
        )
        collect_all_modules_recursively(actual_root_module_to_process)
    else:
        logging.error(
            "Root package object is not a Module or could not be resolved to "
            "a Module. Cannot collect modules."
        )
        return

    final_data = {"modules": sorted(all_modules_data, key=lambda x: x["path"])}

    # Ensure the output directory exists.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)
        logging.info(f"Documentation data successfully written to: {OUTPUT_PATH}")
    except OSError as e:
        logging.error(f"Failed to write JSON output to {OUTPUT_PATH}: {e}")
    except TypeError as e:
        logging.error(f"JSON serialization error: {e}. Check data structures.")
        logging.exception("Serialization error details:")


if __name__ == "__main__":
    main()

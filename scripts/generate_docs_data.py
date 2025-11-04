"""Generates structured documentation data from Python source files.

This script uses griffe to parse a Python package and extract comprehensive
information about modules, classes, functions, and their docstrings. The output
is serialized to JSON for consumption by frontend applications.
"""

import json
import logging
import pathlib
from typing import NotRequired, TypedDict

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

get_logger().setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

PACKAGE_PATH = pathlib.Path("src/lean_explore")
OUTPUT_PATH = pathlib.Path("data/module_data.json")


# --- Types ---


class ParameterDict(TypedDict):
    """Parameter information from function signature and docstring."""

    name: str
    annotation: str
    kind: str
    default: str | None
    description: NotRequired[str]


class ReturnDict(TypedDict):
    """Return value information from function signature and docstring."""

    name: NotRequired[str]
    annotation: str
    description: str


class DocstringAttributeDict(TypedDict):
    """Attribute information from docstring only."""

    name: str
    annotation: str
    description: str
    value: NotRequired[str | None]


class AttributeDict(TypedDict):
    """Attribute information from class code definition."""

    name: str
    value: str | None
    annotation: str
    docstring: str
    path: str
    filepath: str | None
    lineno: int | None


class ExceptionDict(TypedDict):
    """Exception information from docstring raises section."""

    type: str
    description: str


class ExampleDict(TypedDict):
    """Example code from docstring examples section."""

    title: str | None
    code: str


class AdmonitionDict(TypedDict):
    """Admonition (note, warning, etc.) from docstring."""

    title: str
    text: str


class DecoratorDict(TypedDict):
    """Decorator information from function or class definition."""

    text: str
    path: str
    lineno: int | None
    endlineno: int | None


class DocstringSections(TypedDict, total=False):
    """All possible sections parsed from a docstring."""

    summary: str
    text: str
    parameters: list[ParameterDict]
    returns: ReturnDict | list[ReturnDict]
    attributes: list[DocstringAttributeDict]
    raises: list[ExceptionDict]
    examples: list[ExampleDict]
    note: list[AdmonitionDict]
    warning: list[AdmonitionDict]
    deprecated: list[str] | str
    warns: list[str] | str
    yields: list[str] | str
    receives: list[str] | str


class FunctionDict(TypedDict):
    """Serialized function with full documentation."""

    name: str
    path: str
    docstring: str
    docstring_sections: DocstringSections
    parameters: list[ParameterDict]
    returns: ReturnDict
    decorators: list[DecoratorDict]
    is_async: bool
    filepath: str | None
    lineno: int | None
    lines: list[int]


class ClassDict(TypedDict):
    """Serialized class with full documentation."""

    name: str
    path: str
    docstring: str
    docstring_sections: DocstringSections
    methods: list[FunctionDict]
    attributes: list[AttributeDict]
    bases: list[str]
    filepath: str | None
    lineno: int | None
    lines: list[int]


class ModuleDict(TypedDict):
    """Serialized module with full documentation."""

    name: str
    path: str
    filepath: str | None
    docstring: str
    docstring_sections: DocstringSections
    functions: list[FunctionDict]
    classes: list[ClassDict]
    lineno: int | None


ReturnsSectionData = ReturnDict | list[ReturnDict] | None
"""Return type for docstring returns section: single return, multiple, or none."""


def resolve_annotation(annotation: str | Expr | None) -> str:
    """Converts a griffe annotation to its string representation."""
    if isinstance(annotation, (Expr, str)):
        return str(annotation)
    return ""


def extract_summary_and_text(
    sections: list, summary_holder: list[str], text_parts: list[str]
) -> None:
    """Extracts summary and full text from docstring text sections.

    Args:
        sections: List of docstring sections to process.
        summary_holder: Single-element list to store the summary (first text block).
        text_parts: List to accumulate all text parts.
    """
    for section in sections:
        if isinstance(section, DocstringSectionText):
            if not summary_holder[0]:
                summary_holder[0] = section.value.strip()
            text_parts.append(section.value.strip())


def parse_parameters_section(
    section: DocstringSectionParameters,
) -> list[ParameterDict]:
    """Parses a parameters section from a docstring."""
    return [
        {
            "name": parameter.name,
            "annotation": resolve_annotation(parameter.annotation),
            "description": parameter.description.strip()
            if parameter.description
            else "",
            "value": str(parameter.value) if parameter.value is not None else None,
        }
        for parameter in section.value
    ]


def parse_returns_section(section: DocstringSectionReturns) -> ReturnsSectionData:
    """Parses a returns section from a docstring.

    Returns a single dict for one return value, a list for multiple, or None for empty.
    """
    returns_data = [
        {
            "name": item.name if hasattr(item, "name") else "",
            "annotation": resolve_annotation(item.annotation),
            "description": item.description.strip() if item.description else "",
        }
        for item in section.value
    ]

    if len(returns_data) == 1:
        return returns_data[0]
    elif len(returns_data) > 1:
        return returns_data
    return None


def parse_attributes_section(
    section: DocstringSectionAttributes,
) -> list[DocstringAttributeDict]:
    """Parses an attributes section from a docstring."""
    return [
        {
            "name": attribute.name,
            "annotation": resolve_annotation(attribute.annotation),
            "description": attribute.description.strip()
            if attribute.description
            else "",
        }
        for attribute in section.value
    ]


def parse_raises_section(section: DocstringSectionRaises) -> list[ExceptionDict]:
    """Parses a raises section from a docstring."""
    return [
        {
            "type": resolve_annotation(exception.annotation),
            "description": exception.description.strip()
            if exception.description
            else "",
        }
        for exception in section.value
    ]


def parse_examples_section(section: DocstringSectionExamples) -> list[ExampleDict]:
    """Parses an examples section from a docstring."""
    return [
        {
            "title": example.title.strip() if example.title else None,
            "code": example.value.strip(),
        }
        for example in section.value
    ]


def parse_admonition_section(
    section: DocstringSectionAdmonition, sections_data: DocstringSections
) -> None:
    """Parses an admonition section (note, warning, etc.).

    Adds the admonition to sections_data.
    """
    payload = section.value

    # Validate payload structure before accessing attributes
    if not (hasattr(payload, "kind") and isinstance(payload.kind, str)):
        return
    if not (hasattr(payload, "text") and isinstance(payload.text, str)):
        return

    kind = payload.kind
    admonition_item = {
        "title": section.title.strip() if section.title else kind,
        "text": payload.text.strip(),
    }

    if kind not in sections_data:
        sections_data[kind] = []
    sections_data[kind].append(admonition_item)


def parse_generic_section(section) -> list | str:
    """Parses generic docstring sections like warns, yields, etc."""
    if not hasattr(section, "value") or not isinstance(section.value, list):
        return (
            str(section.value)
            if hasattr(section, "value")
            else "Unsupported section structure"
        )

    values = []
    for item in section.value:
        if isinstance(item, DocstringNamedElement):
            values.append(
                {
                    "name": item.name,
                    "annotation": resolve_annotation(item.annotation)
                    if hasattr(item, "annotation")
                    else "",
                    "description": item.description.strip() if item.description else "",
                }
            )
        elif hasattr(item, "name") and hasattr(item, "description"):
            values.append(
                {
                    "name": item.name,
                    "description": item.description.strip() if item.description else "",
                }
            )
        elif hasattr(item, "text"):
            values.append(item.text.strip())
        else:
            values.append(str(item))
    return values


def parse_docstring(docstring_object: object | None) -> DocstringSections:
    """Parses all sections from a griffe docstring object into structured data."""
    if not docstring_object or not hasattr(docstring_object, "parsed"):
        return DocstringSections()

    sections_data: DocstringSections = DocstringSections()
    summary = [""]
    text_parts = []

    # First pass: extract summary and text
    extract_summary_and_text(docstring_object.parsed, summary, text_parts)

    # Second pass: parse all section types
    for section in docstring_object.parsed:
        kind = section.kind.value

        if isinstance(section, DocstringSectionText):
            continue  # Already handled in first pass
        elif isinstance(section, DocstringSectionParameters):
            sections_data[kind] = parse_parameters_section(section)
        elif isinstance(section, DocstringSectionReturns):
            result = parse_returns_section(section)
            if result:
                sections_data[kind] = result
        elif isinstance(section, DocstringSectionAttributes):
            sections_data[kind] = parse_attributes_section(section)
        elif isinstance(section, DocstringSectionRaises):
            sections_data[kind] = parse_raises_section(section)
        elif isinstance(section, DocstringSectionExamples):
            sections_data[kind] = parse_examples_section(section)
        elif isinstance(section, DocstringSectionAdmonition):
            parse_admonition_section(section, sections_data)
        elif isinstance(
            section,
            (
                DocstringSectionDeprecated,
                DocstringSectionWarns,
                DocstringSectionYields,
                DocstringSectionReceives,
                DocstringSectionOtherParameters,
                DocstringSectionClasses,
                DocstringSectionFunctions,
                DocstringSectionModules,
            ),
        ):
            sections_data[kind] = parse_generic_section(section)
        else:
            sections_data[kind] = parse_generic_section(section)

    # Add summary and consolidated text
    if summary[0]:
        sections_data["summary"] = summary[0]

    text_content = "\n\n".join(part for part in text_parts if part)
    if summary[0] and text_content.strip().startswith(summary[0].strip()):
        sections_data["text"] = text_content
    elif summary[0]:
        sections_data["text"] = (
            f"{summary[0]}\n\n{text_content}".strip() if text_content else summary[0]
        )
    else:
        sections_data["text"] = text_content

    return sections_data


def serialize_typer_option(function_call: ExprCall) -> str:
    """Formats a typer.Option call as a multi-line string."""
    function_name = str(function_call.function)
    arguments: list[str] = []

    if hasattr(function_call, "arguments"):
        arguments = [str(argument) for argument in function_call.arguments]

    if not arguments:
        return f"{function_name}()"

    indent = "    "
    formatted_arguments = f",\n{indent}".join(arguments)
    return f"{function_name}(\n{indent}{formatted_arguments}\n)"


def serialize_parameters(parameters: Parameters) -> list[ParameterDict]:
    """Converts griffe Parameters into serializable dictionaries."""
    if not parameters:
        return []

    result: list[ParameterDict] = []
    for parameter in parameters:
        default_representation: str | None = None

        if parameter.default is not None:
            # Special formatting for typer.Option calls
            if (
                isinstance(parameter.default, ExprCall)
                and hasattr(parameter.default, "function")
                and str(parameter.default.function) == "typer.Option"
            ):
                default_representation = serialize_typer_option(parameter.default)
            else:
                default_representation = str(parameter.default)

        result.append(
            {
                "name": parameter.name,
                "annotation": resolve_annotation(parameter.annotation),
                "kind": parameter.kind.value,
                "default": default_representation,
            }
        )

    return result


def serialize_decorators(decorators: list[Decorator]) -> list[DecoratorDict]:
    """Converts griffe Decorator objects into serializable dictionaries."""
    return [
        {
            "text": str(decorator.value),
            "path": decorator.callable_path,
            "lineno": getattr(decorator, "lineno", None),
            "endlineno": getattr(decorator, "endlineno", None),
        }
        for decorator in decorators
    ]


def merge_parameter_descriptions(
    code_parameters: list[ParameterDict], docstring_parameters: list[ParameterDict]
) -> None:
    """Merges docstring descriptions into code parameters in-place."""
    docstring_map = {parameter["name"]: parameter for parameter in docstring_parameters}

    for code_parameter in code_parameters:
        if code_parameter["name"] in docstring_map:
            code_parameter["description"] = docstring_map[code_parameter["name"]].get(
                "description", ""
            )


def build_returns_info(
    function: Function, docstring_sections: DocstringSections
) -> ReturnDict:
    """Builds return information combining code annotation and docstring description."""
    returns_info = {
        "annotation": resolve_annotation(function.returns),
        "description": "",
    }

    docstring_returns = docstring_sections.get("returns")
    if isinstance(docstring_returns, dict):
        returns_info["description"] = docstring_returns.get("description", "")
        if docstring_returns.get("annotation"):
            returns_info["annotation"] = docstring_returns.get("annotation")
    elif isinstance(docstring_returns, list) and docstring_returns:
        first_return = docstring_returns[0]
        returns_info["description"] = first_return.get("description", "")
        if first_return.get("annotation"):
            returns_info["annotation"] = first_return.get("annotation")
        if len(docstring_returns) > 1:
            returns_info["description"] += " (Multiple return paths documented)"

    return returns_info


def serialize_function(function: Function, module_path: str) -> FunctionDict:
    """Converts a griffe Function into a serializable dictionary.

    Includes full documentation from both code and docstrings.
    """
    docstring_sections = parse_docstring(function.docstring)
    code_parameters = serialize_parameters(function.parameters)

    # Merge docstring parameter descriptions into code parameters
    merge_parameter_descriptions(
        code_parameters, docstring_sections.get("parameters", [])
    )

    return {
        "name": function.name,
        "path": function.canonical_path,
        "docstring": function.docstring.value if function.docstring else "",
        "docstring_sections": docstring_sections,
        "parameters": code_parameters,
        "returns": build_returns_info(function, docstring_sections),
        "decorators": serialize_decorators(function.decorators),
        "is_async": getattr(function, "is_async", False),
        "filepath": str(function.filepath.relative_to(pathlib.Path.cwd()))
        if function.filepath
        else None,
        "lineno": function.lineno,
        "lines": [function.lineno, function.endlineno]
        if function.lineno and function.endlineno
        else [],
    }


def get_definition_module_path(object: Function | Class | Module) -> str:
    """Determines the canonical module path where an object is defined."""
    if (
        hasattr(object, "parent")
        and object.parent
        and hasattr(object.parent, "canonical_path")
    ):
        return object.parent.canonical_path
    elif "." in object.canonical_path:
        return object.canonical_path.rsplit(".", 1)[0]
    else:
        return object.canonical_path


def serialize_module(module: Module) -> ModuleDict:
    """Converts a griffe Module into a serializable dictionary.

    Only includes functions and classes defined directly in this module,
    not imported ones.
    """
    functions = []
    classes = []
    current_path = module.canonical_path

    for member_name, member in module.members.items():
        try:
            # Resolve aliases to their targets
            target = member.final_target if member.is_alias else member

            if not target or not isinstance(target.canonical_path, str):
                continue

            definition_path = get_definition_module_path(target)

            # Only include items defined in this module
            if target.is_function and isinstance(target, Function):
                if definition_path == current_path:
                    functions.append(serialize_function(target, current_path))
            elif target.is_class and isinstance(target, Class):
                if definition_path == current_path:
                    classes.append(serialize_class(target, current_path))

        except AliasResolutionError:
            # External imports cannot be resolved, skip them
            continue

    return {
        "name": module.name,
        "path": module.canonical_path,
        "filepath": str(module.filepath.relative_to(pathlib.Path.cwd()))
        if module.filepath
        else None,
        "docstring": module.docstring.value if module.docstring else "",
        "docstring_sections": parse_docstring(module.docstring),
        "functions": sorted(functions, key=lambda x: x["name"]),
        "classes": sorted(classes, key=lambda x: x["name"]),
        "lineno": module.lineno,
    }


def merge_docstring_attributes(
    code_attributes: list[AttributeDict],
    docstring_attributes: list[DocstringAttributeDict],
    class_path: str,
) -> None:
    """Merges docstring-only attributes with code attributes.

    Adds attributes that only appear in docstrings, and fills in
    missing docstrings for code attributes.
    """
    existing_names = {attribute["name"] for attribute in code_attributes}

    for docstring_attribute in docstring_attributes:
        name = docstring_attribute["name"]

        if name not in existing_names:
            # Add attribute that only exists in docstring
            code_attributes.append(
                {
                    "name": name,
                    "value": None,
                    "annotation": docstring_attribute.get("annotation", ""),
                    "docstring": docstring_attribute.get("description", ""),
                    "path": f"{class_path}.{name}",
                    "filepath": None,
                    "lineno": None,
                }
            )
        else:
            # Fill in docstring for existing code attribute
            for code_attribute in code_attributes:
                if code_attribute["name"] == name and not code_attribute["docstring"]:
                    code_attribute["docstring"] = docstring_attribute.get(
                        "description", ""
                    )
                    break


def serialize_class(class_object: Class, module_path: str) -> ClassDict:
    """Converts a griffe Class into a serializable dictionary.

    Includes full documentation from both code and docstrings.
    """
    methods = []
    attributes = []

    for member_name, member in class_object.members.items():
        if member.is_attribute:
            attributes.append(
                {
                    "name": member.name,
                    "value": str(member.value) if member.value is not None else None,
                    "annotation": resolve_annotation(member.annotation),
                    "docstring": member.docstring.value if member.docstring else "",
                    "path": member.canonical_path,
                    "filepath": str(member.filepath.relative_to(pathlib.Path.cwd()))
                    if member.filepath
                    else None,
                    "lineno": member.lineno,
                }
            )
        elif member.is_function:
            try:
                actual_method = member.final_target if member.is_alias else member
                if actual_method:
                    methods.append(
                        serialize_function(actual_method, class_object.canonical_path)
                    )
            except AliasResolutionError:
                # External imports cannot be resolved, skip them
                continue

    docstring_sections = parse_docstring(class_object.docstring)
    merge_docstring_attributes(
        attributes,
        docstring_sections.get("attributes", []),
        class_object.canonical_path,
    )

    return {
        "name": class_object.name,
        "path": class_object.canonical_path,
        "docstring": class_object.docstring.value if class_object.docstring else "",
        "docstring_sections": docstring_sections,
        "methods": sorted(methods, key=lambda x: (x["name"] != "__init__", x["name"])),
        "attributes": sorted(attributes, key=lambda x: x["name"]),
        "bases": [resolve_annotation(base) for base in class_object.bases],
        "filepath": str(class_object.filepath.relative_to(pathlib.Path.cwd()))
        if class_object.filepath
        else None,
        "lineno": class_object.lineno,
        "lines": [class_object.lineno, class_object.endlineno]
        if class_object.lineno and class_object.endlineno
        else [],
    }


def is_target_package_module(module: Module, package_name: str) -> bool:
    """Checks if a module belongs to the target package.

    Uses both filepath and canonical name to determine membership.
    """
    # Check by filepath
    if module.filepath:
        absolute_module_path = module.filepath.resolve()
        absolute_package_path = PACKAGE_PATH.resolve()
        if (
            absolute_package_path == absolute_module_path
            or absolute_package_path in absolute_module_path.parents
        ):
            return True

    # Check by canonical name
    canonical_name = module.canonical_path
    if (
        package_name
        and isinstance(canonical_name, str)
        and canonical_name.startswith(package_name)
    ):
        return True

    return False


def collect_modules_recursively(
    module: Module, package_name: str, processed: set[str]
) -> list[ModuleDict]:
    """Recursively collects and serializes all modules in the package.

    Args:
        module: The current module to process.
        package_name: Name of the root package.
        processed: Set of already processed module paths to avoid duplicates.

    Returns:
        List of serialized module dictionaries.
    """
    if (
        not module
        or not hasattr(module, "canonical_path")
        or module.canonical_path in processed
    ):
        return []

    if not is_target_package_module(module, package_name):
        return []

    logging.info(f"Processing module: {module.canonical_path}")
    processed.add(module.canonical_path)

    modules = [serialize_module(module)]

    # Recursively process submodules
    for member in module.members.values():
        try:
            if member.is_module:
                actual_member = member.final_target if member.is_alias else member
                if actual_member:
                    modules.extend(
                        collect_modules_recursively(
                            actual_member, package_name, processed
                        )
                    )
        except AliasResolutionError:
            # External module imports cannot be resolved, skip them
            continue

    return modules


def main() -> None:
    """Generates documentation data from Python package and writes to JSON file."""
    logging.info(f"Starting documentation generation for package: {PACKAGE_PATH}")

    loader = GriffeLoader(
        search_paths=[str(PACKAGE_PATH.parent)], docstring_parser="google"
    )
    package_name = PACKAGE_PATH.name
    root_package = loader.load(package_name)
    loader.resolve_aliases(implicit=True, external=False)

    # Resolve root module from loaded package
    if isinstance(root_package, Module):
        root_module = root_package
    elif isinstance(root_package, Alias) and isinstance(
        root_package.resolved_target, Module
    ):
        root_module = root_package.resolved_target
    else:
        raise ValueError(f"Failed to resolve root module for package: {package_name}")

    logging.info(f"Collecting modules from root: {root_module.canonical_path}")
    processed_paths: set[str] = set()
    modules = collect_modules_recursively(root_module, package_name, processed_paths)

    output_data = {"modules": sorted(modules, key=lambda x: x["path"])}

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(output_data, file, indent=2, ensure_ascii=False)

    logging.info(f"Documentation data successfully written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

from typing import (
    Any,
    Type,
    List,
    Dict,
)
from pydantic import BaseModel


def json_interface_string(model_type: Type[BaseModel]) -> str:
    """
    Generates a TypeScript-style interface string for a given Pydantic model.

    Args:
        model_type: The Pydantic model class.

    Returns:
        A string representing the TypeScript-style JSON interface.
    """
    json_schema = model_type.model_json_schema()

    return _json_interface_string(json_schema, root_schema=json_schema)


def _json_interface_string(
    current_schema: Dict[str, Any], root_schema: Dict[str, Any], indent: int = 1
) -> str:
    """
    Generates a TypeScript-style interface string for a given Pydantic model.

    Args:
        current_schema: The current part of the JSON schema being processed.
        root_schema: The root of the entire JSON schema, used for resolving $refs.
        indent: The current indentation level.

    Returns:
        A string representing the TypeScript-style JSON interface.
    """
    ret_str: str = "{"

    # Helper for indentation and new lines
    def ind_nl(n: int) -> str:
        return "\n" + "\t" * n

    # Properties to iterate over. Could be in 'properties' or from 'allOf' etc.
    properties_to_process: Dict[str, Any] = {}
    required_fields: List[str] = current_schema.get("required", [])

    if "properties" in current_schema:
        properties_to_process.update(current_schema["properties"])

    # Basic handling for 'allOf' - merges properties from referenced schemas
    # A more robust solution would deeply merge schemas if necessary.
    if "allOf" in current_schema:
        for item in current_schema["allOf"]:
            if "$ref" in item:
                ref_path = item["$ref"].split("/")
                ref_schema = _resolve_ref(ref_path, root_schema)
                if "properties" in ref_schema:
                    properties_to_process.update(ref_schema["properties"])
                if "required" in ref_schema:  # Collect required fields from allOf parts
                    required_fields.extend(
                        r for r in ref_schema["required"] if r not in required_fields
                    )

    if not properties_to_process:
        # In the case of an "empty object" or a schema with no properties
        # just assume it can be represented by "{}"
        # TODO: Decide if this is the desired behavior.
        return "{}"

    for key, value_schema in properties_to_process.items():
        ret_str += ind_nl(indent) + f"{key}"

        is_required = key in required_fields
        ret_str += " ?: " if not is_required else " : "

        ret_str += (
            f"{_typescript_type_from_schema(value_schema, root_schema, indent + 1)};"
        )

        if "description" in value_schema:
            ret_str += f" // {value_schema['description']}"

    ret_str += ind_nl(indent - 1) + "}"
    return ret_str


def _resolve_ref(ref_path: List[str], root_schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolves a $ref path against the root schema.
    Example $ref: "#/$defs/MyModel" -> path: ["#", "$defs", "MyModel"]
    """
    if not ref_path or ref_path[0] != "#":
        raise ValueError(f"Invalid $ref format: {ref_path}")

    current_level = root_schema
    for part in ref_path[1:]:  # Skip "#"
        if part not in current_level:
            raise ValueError(
                f"Reference part '{part}' not found in schema for $ref {'/'.join(ref_path)}"
            )
        current_level = current_level[part]
    return current_level


def _typescript_type_from_schema(
    value_schema: Dict[str, Any], root_schema: Dict[str, Any], indent: int
) -> str:
    """
    Determines the TypeScript type string for a given JSON schema definition.

    Args:
        value_schema: The JSON schema for the specific value/property.
        root_schema: The root of the entire JSON schema, for $ref resolution.
        indent: Current indentation level (for nested objects).

    Returns:
        TypeScript type string.
    """
    if "$ref" in value_schema:
        ref_path = value_schema["$ref"].split("/")
        # Find the referenced schema in the root schema
        resolved_schema = _resolve_ref(ref_path, root_schema)
        # Recursively resolve the type from the referenced schema
        return _typescript_type_from_schema(resolved_schema, root_schema, indent)

    # Handle 'anyOf' for nullable types (e.g., string | null) or other unions
    if "anyOf" in value_schema:
        types = []
        for sub_schema in value_schema["anyOf"]:
            types.append(_typescript_type_from_schema(sub_schema, root_schema, indent))
        # Filter out duplicates and join (sorted for deterministic output)
        return " | ".join(sorted(list(set(types))))

    # Handle 'enum' from Pydantic Literal/Enum (JSON schema 'enum' or 'const')
    if "enum" in value_schema:
        return " | ".join(
            [f'"{v}"' if isinstance(v, str) else str(v) for v in value_schema["enum"]]
        )
    if "const" in value_schema:  # For Literal['a'] (why would we do this though!?)
        const_val = value_schema["const"]
        return f'"{const_val}"' if isinstance(const_val, str) else str(const_val)

    schema_type = value_schema.get("type")

    if isinstance(schema_type, list):
        ts_types = []
        for t in schema_type:
            if t == "null":
                ts_types.append("null")
            else:
                # Create a temporary schema for the non-null type to recursively call
                temp_schema_for_type = value_schema.copy()
                temp_schema_for_type["type"] = t
                # Remove array specific 'items' if we are just getting type for one of list elements
                if "items" in temp_schema_for_type and t != "array":
                    del temp_schema_for_type["items"]
                ts_types.append(
                    _typescript_type_from_schema(
                        temp_schema_for_type, root_schema, indent
                    )
                )
        return " | ".join(sorted(list(set(ts_types))))

    if schema_type == "array":
        if "items" not in value_schema:
            # Default to any[] if items are not specified (although this is very unusual)
            # TODO: Decide if this is the desired behavior.
            return "any[]"
        items_schema = value_schema["items"]
        item_type_str = _typescript_type_from_schema(items_schema, root_schema, indent)
        # Handle cases where item_type_str itself might be a union type (e.g., (string | number)[])
        if " | " in item_type_str:
            return f"({item_type_str})[]"
        return f"{item_type_str}[]"

    elif schema_type == "object":
        # This is for inline object definitions (not $ref)
        # Check if it has properties or additionalProperties to determine structure
        if "properties" in value_schema or "additionalProperties" in value_schema:
            return _json_interface_string(value_schema, root_schema, indent)
        else:
            # An object type with no specified properties, could be a generic object
            # TODO: Decide if this is the desired behavior.
            return "object"
    else:
        # Otherwise, we just assume it's a primitive type (or if not specified, "any")
        # TODO: Decide if this is the desired behavior.
        if schema_type == "string" and value_schema.get("pattern") is not None:
            # Handle regex patterns for strings
            # NOTE: Is this maybe excessive and confusing for Agents?
            return f'string /* regex: "{value_schema["pattern"]}" */'
        return schema_type if schema_type else "any"

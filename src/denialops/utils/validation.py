"""Schema validation utilities."""

import json
from pathlib import Path
from typing import Any

import jsonschema

# Schema directory relative to this file
SCHEMAS_DIR = Path(__file__).parent.parent.parent.parent / "schemas"


def _load_schema(schema_name: str) -> dict[str, Any]:
    """Load a JSON schema by name."""
    schema_path = SCHEMAS_DIR / f"{schema_name}.schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema not found: {schema_path}")
    return json.loads(schema_path.read_text())


def validate_case_facts(data: dict[str, Any]) -> list[str]:
    """
    Validate case facts against schema.

    Returns list of validation errors (empty if valid).
    """
    try:
        schema = _load_schema("case_facts")
        jsonschema.validate(instance=data, schema=schema)
        return []
    except jsonschema.ValidationError as e:
        return [str(e.message)]
    except FileNotFoundError as e:
        return [str(e)]


def validate_route(data: dict[str, Any]) -> list[str]:
    """
    Validate route decision against schema.

    Returns list of validation errors (empty if valid).
    """
    try:
        schema = _load_schema("route")
        jsonschema.validate(instance=data, schema=schema)
        return []
    except jsonschema.ValidationError as e:
        return [str(e.message)]
    except FileNotFoundError as e:
        return [str(e)]


def validate_action_plan(data: dict[str, Any]) -> list[str]:
    """
    Validate action plan against schema.

    Returns list of validation errors (empty if valid).
    """
    try:
        schema = _load_schema("action_plan")
        jsonschema.validate(instance=data, schema=schema)
        return []
    except jsonschema.ValidationError as e:
        return [str(e.message)]
    except FileNotFoundError as e:
        return [str(e)]


def validate_plan_rules(data: dict[str, Any]) -> list[str]:
    """
    Validate plan rules against schema.

    Returns list of validation errors (empty if valid).
    """
    try:
        schema = _load_schema("plan_rules")
        jsonschema.validate(instance=data, schema=schema)
        return []
    except jsonschema.ValidationError as e:
        return [str(e.message)]
    except FileNotFoundError as e:
        return [str(e)]

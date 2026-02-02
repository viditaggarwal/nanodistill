"""Schema validation and field filtering utilities for Pydantic models.

Provides functions for:
- Extracting valid field names from Pydantic models
- Filtering extra fields from generated examples
- Validating data against Pydantic schemas with comprehensive error reporting
"""

import logging
from typing import Any, Dict, Optional, Set, Tuple, Type

from pydantic import BaseModel, ValidationError


def get_model_field_names(response_model: Type[BaseModel]) -> Set[str]:
    """Extract valid field names from a Pydantic model.

    Args:
        response_model: Pydantic model class

    Returns:
        Set of field names defined in the model
    """
    return set(response_model.model_fields.keys())


def filter_extra_fields(
    data: Dict[str, Any],
    response_model: Type[BaseModel],
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Filter dictionary to only include fields in Pydantic model.

    Removes any fields not defined in the Pydantic schema.
    Logs removed fields with counts.

    Args:
        data: Dictionary to filter
        response_model: Pydantic model defining valid fields
        logger: Optional logger for tracking removed fields

    Returns:
        Filtered dict with only valid fields
    """
    valid_fields = get_model_field_names(response_model)
    filtered = {}
    removed_fields = []

    for key, value in data.items():
        if key in valid_fields:
            filtered[key] = value
        else:
            removed_fields.append(key)

    if removed_fields and logger:
        logger.info(f"Filtered {len(removed_fields)} extra fields: {', '.join(removed_fields)}")

    return filtered


def validate_against_schema(
    data: Dict[str, Any],
    response_model: Type[BaseModel],
    logger: Optional[logging.Logger] = None,
) -> Tuple[bool, Optional[str]]:
    """Validate dictionary against Pydantic schema.

    Args:
        data: Dictionary to validate
        response_model: Pydantic model to validate against
        logger: Optional logger for validation errors

    Returns:
        Tuple of (is_valid, error_message):
        - is_valid: True if validation passed
        - error_message: None if valid, error string if invalid
    """
    try:
        # Try to construct model instance to validate
        response_model.model_validate(data)
        return True, None
    except ValidationError as e:
        error_msg = str(e)
        if logger:
            logger.debug(f"Validation error: {error_msg}")
        return False, error_msg
    except Exception as e:
        error_msg = f"Unexpected validation error: {str(e)}"
        if logger:
            logger.debug(error_msg)
        return False, error_msg

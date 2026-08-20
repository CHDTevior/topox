"""KTJD-17 lossless-v1 data contracts.

The package is intentionally independent from the legacy AnyTop13 loader.  A
KTJD-17 artifact must be validated under its own schema before later parser,
converter, loader, or model code is allowed to consume it.
"""

from .schema import (  # noqa: F401
    KTJD17_D,
    KTJD17_REPR_VERSION,
    SchemaValidationError,
    build_schema,
    load_schema,
    validate_physical_parent_tree,
    validate_schema,
    validate_unit_metadata,
    write_schema,
)

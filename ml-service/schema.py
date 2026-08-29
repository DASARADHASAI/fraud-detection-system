"""
=============================================================================
PHASE 14 -- THE DATA BOUNDARY. Map anyone's columns onto our canonical names.
=============================================================================

WHY THIS FILE EXISTS:
Every other file in this service speaks one vocabulary -- `amount`,
`nameOrig`, `oldbalanceOrg` and so on. Those happen to be PaySim's column
names, and they appear about 480 times across 13 files.

Rather than make all 480 references indirect, we rename the USER's columns to
ours once, at the single moment the file is read. Downstream code never learns
that anything changed. This is the standard "normalise at the boundary"
pattern, and it is why supporting other datasets is one module rather than a
rewrite.

>>> A MAPPING MISTAKE IS SILENT <<<
If `amount` is wired to the wrong column, every score is computed from the
wrong evidence and NOTHING raises an exception. So this module fails loudly
and early: it would rather refuse to start than score confidently from a
column it guessed at.
=============================================================================
"""

from __future__ import annotations


class SchemaError(ValueError):
    """Raised when the column mapping cannot be used. Carries every problem
    found, not just the first -- fixing config one error per run is miserable."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("; ".join(errors))


# --- The canonical vocabulary -------------------------------------------------
#
# REQUIRED means the pipeline cannot run without it. OPTIONAL means it can run,
# but a whole feature group switches off -- and the capability report says so.

REQUIRED_CANONICAL = ["step", "type", "amount", "nameOrig", "nameDest"]

OPTIONAL_CANONICAL = [
    "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
]

# The ground-truth label. Needed to TRAIN, absent when merely scoring.
LABEL_CANONICAL = "isFraud"

CANONICAL_FIELDS = REQUIRED_CANONICAL + OPTIONAL_CANONICAL + [LABEL_CANONICAL]

# Columns that must be numeric. A string "1500.00" in `amount` would compare
# and sum in ways that look almost right, which is worse than crashing.
NUMERIC_CANONICAL = [
    "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
]


def load_mapping(cfg: dict) -> dict[str, str]:
    """
    Read the `schema:` block and return canonical name -> the user's column name.

    Anything the user does not mention maps to itself, so a PaySim file needs no
    schema block at all and behaves exactly as it always has.
    """
    raw = (cfg or {}).get("schema") or {}
    errors: list[str] = []

    unknown = [k for k in raw if k not in CANONICAL_FIELDS]
    for name in unknown:
        errors.append(
            f"'{name}' is not a canonical field name. Valid names: "
            f"{', '.join(CANONICAL_FIELDS)}"
        )

    mapping: dict[str, str] = {}
    for field in CANONICAL_FIELDS:
        source = raw.get(field, field)
        if source is None:
            if field in REQUIRED_CANONICAL:
                errors.append(
                    f"'{field}' is required and cannot be mapped to null. "
                    f"Every transaction file must identify a {field}."
                )
            continue
        mapping[field] = str(source)

    # Two canonical fields reading the same source column is always a mistake,
    # and it produces a DataFrame with duplicate column names downstream.
    seen: dict[str, str] = {}
    for field, source in mapping.items():
        if source in seen:
            errors.append(
                f"both '{seen[source]}' and '{field}' are mapped to the same "
                f"column '{source}'"
            )
        else:
            seen[source] = field

    if errors:
        raise SchemaError(errors)
    return mapping

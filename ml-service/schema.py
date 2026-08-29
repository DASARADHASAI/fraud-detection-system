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

import pandas as pd


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


class CapabilityReport:
    """
    What the supplied data can and cannot support.

    >>> WHY THIS EXISTS <<<
    Missing balance columns do not crash anything. Eleven features simply become
    constant zero, the forest never splits on them, and you have silently trained
    a 17-feature model believing it is a 28-feature one. On PaySim those eleven
    carried 68.3% of model importance, and losing them took fraud blocking from
    149/150 to 1/150. A degradation that large must not be silent.
    """

    def __init__(self, mapped: list[str], absent: list[str]) -> None:
        self.mapped = mapped
        self.absent = absent
        self.groups: dict[str, bool] = {}          # filled by the resolver
        self.active_feature_count: int = 0         # filled by the resolver

    def render(self) -> str:
        lines = [
            "DATA CAPABILITY REPORT",
            f"  Mapped        {', '.join(self.mapped) or '(none)'}",
            f"  Not supplied  {', '.join(self.absent) or '(none)'}",
        ]
        if self.groups:
            lines.append("")
            lines.append("  Feature groups")
            for name, on in self.groups.items():
                lines.append(f"    {name:<24} {'ON ' if on else 'OFF'}")
            lines.append(f"  Active feature count  {self.active_feature_count}")

        balance_absent = [f for f in self.absent if f in OPTIONAL_CANONICAL]
        if balance_absent:
            lines += [
                "",
                "  COST OF THE MISSING BALANCE COLUMNS",
                f"    {', '.join(balance_absent)}",
                "    On PaySim these carried 68.3% of model importance.",
                "    Balance-error signals will be inert; account-drain",
                "    detection is unavailable.",
                "    Encoded as 'not recorded' (zero), NOT as reconciling",
                "    balances -- a reconciling balance is the fraud signature",
                "    in 99.5% of cases.",
            ]
        return "\n".join(lines)


def normalize(df: pd.DataFrame, mapping: dict[str, str]):
    """
    Rename the user's columns to canonical names, coerce numerics, and report
    which optional fields were not supplied.

    Returns (renamed_frame, CapabilityReport).

    >>> WHY THE ERROR NAMES BOTH NAMES <<<
    A user reading "missing column 'amount'" will search their CSV for
    "amount", not find it, and be stuck -- because in THEIR file it is called
    "amt". The error has to speak both vocabularies.
    """
    errors: list[str] = []
    present = set(df.columns)

    for field in REQUIRED_CANONICAL:
        source = mapping[field]
        if source not in present:
            errors.append(
                f"required field '{field}' is mapped to column '{source}', "
                f"which is not in the file. Columns found: {sorted(present)}"
            )
    if errors:
        raise SchemaError(errors)

    rename = {mapping[f]: f for f in mapping if mapping[f] in present}
    out = df.rename(columns=rename).copy()

    absent = [f for f in CANONICAL_FIELDS
              if f not in mapping or mapping[f] not in present]

    for field in NUMERIC_CANONICAL:
        if field in out.columns:
            # errors="coerce" turns unparseable junk into NaN rather than
            # throwing. NaN is then visible and testable; a silent string is not.
            out[field] = pd.to_numeric(out[field], errors="coerce").astype(float)

    # An unsupplied balance becomes PaySim's own "not recorded" encoding: zero.
    # NEVER a reconciling balance -- see the CapabilityReport docstring.
    for field in OPTIONAL_CANONICAL:
        if field not in out.columns:
            out[field] = 0.0

    mapped = [f for f in CANONICAL_FIELDS if f not in absent]
    return out, CapabilityReport(mapped=mapped, absent=absent)

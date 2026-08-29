# Bring Your Own Data — Phase 14 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the pipeline read a transaction file whose columns are named
anything, by normalising to canonical names at a single boundary, and report
honestly which feature groups the supplied data can support.

**Architecture:** One new module, `ml-service/schema.py`, renames the user's
columns to the canonical vocabulary the moment the CSV is read, inside
`features.py::load_raw_data()`. Every one of the ~480 downstream references to
PaySim column names keeps working untouched. `MODEL_FEATURES` becomes the output
of a resolver over four feature groups, each switched on by what the data
supports. The default mapping is identity, so PaySim behaviour is unchanged —
which the existing five test suites prove.

**Tech Stack:** Python 3.14, pandas, PyYAML, scikit-learn. Tests are **standalone
scripts**, not pytest — see Global Constraints.

**Spec:** `docs/superpowers/specs/2026-08-29-bring-your-own-data-design.md`

## Global Constraints

- **Tests are standalone scripts.** This repo does NOT use pytest. Every test
  file defines `failures: list[str]`, a `check(condition, description, detail="")`
  helper, and a `main() -> int` that prints `ALL CHECKS PASSED` and returns 0, or
  lists failures and returns 1. Copy the harness from `ml-service/test_upi.py:29-41`.
- **Run tests as:** `cd ml-service && ./venv/bin/python test_schema.py`
- **The identity-mapping regression gate is non-negotiable.** After every task,
  `test_api.py`, `test_upi.py`, `test_webhook.py`, `test_explain.py` and
  `test_genai.py` must still pass. If any changes behaviour under identity
  mapping, the boundary normalisation is wrong.
- **Never invent a balance.** An unsupplied balance is encoded `0.0` — PaySim's
  own "not recorded" convention. A reconciling balance is the fraud signature in
  99.5% of cases. See spec §4a.
- **Causality is untouched.** Do not modify `add_account_profiles()` or
  `_add_account_profile()`. They already use an expanding window excluding the
  current row.
- **Comment heavily and explain jargon inline.** This is the repo's house style;
  match `features.py` and `webhook_razorpay.py`.
- **Commit trailer.** Every commit message ends with:
  `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`
- **Do not retrain.** No task in this phase runs `train_model.py` or
  `features.py main()` over the 6.3M-row file.

---

### Task 1: Canonical field table and mapping validation

**Files:**
- Create: `ml-service/schema.py`
- Create: `ml-service/test_schema.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `SchemaError(ValueError)` with `.errors: list[str]`;
  `REQUIRED_CANONICAL: list[str]`, `OPTIONAL_CANONICAL: list[str]`,
  `LABEL_CANONICAL: str`, `CANONICAL_FIELDS: list[str]`,
  `NUMERIC_CANONICAL: list[str]`;
  `load_mapping(cfg: dict) -> dict[str, str]` returning canonical→source.

- [ ] **Step 1: Write the failing test**

Create `ml-service/test_schema.py`:

```python
"""
Tests for the data boundary (Phase 14).

schema.py is the only place that knows what the user's columns are called.
If it maps a column wrongly, every downstream number is computed from the
wrong evidence and nothing raises -- exactly the class of bug that passes
testing. So these checks are deliberately picky about failure messages.

Run:
    ./venv/bin/python test_schema.py
"""

from __future__ import annotations

import sys

import schema

failures: list[str] = []


def check(condition: bool, description: str, detail: str = "") -> None:
    if condition:
        print(f"  PASS  {description}")
    else:
        print(f"  FAIL  {description}")
        if detail:
            print(f"        {detail}")
        failures.append(description)


def rejects(cfg: dict, description: str, expect_in: str) -> None:
    """Assert load_mapping refuses a config, and refuses it for the right reason."""
    try:
        schema.load_mapping(cfg)
    except schema.SchemaError as e:
        joined = "; ".join(e.errors)
        check(expect_in.lower() in joined.lower(), description,
              f"expected {expect_in!r} in: {joined}")
    else:
        check(False, description, "no SchemaError was raised")


def a_identity_default() -> None:
    print("\n1. An empty schema block means identity mapping")
    mapping = schema.load_mapping({})
    for field in schema.CANONICAL_FIELDS:
        check(mapping[field] == field, f"{field} defaults to itself")


def a_explicit_mapping() -> None:
    print("\n2. An explicit mapping is honoured")
    cfg = {"schema": {"amount": "amt", "nameOrig": "cc_num", "nameDest": "merchant"}}
    mapping = schema.load_mapping(cfg)
    check(mapping["amount"] == "amt", "amount -> amt")
    check(mapping["nameOrig"] == "cc_num", "nameOrig -> cc_num")
    check(mapping["step"] == "step", "unlisted fields still default to identity")


def a_rejections() -> None:
    print("\n3. Bad mappings are refused, with a usable reason")
    rejects({"schema": {"nonsense": "x"}},
            "an unknown canonical name is refused", "nonsense")
    rejects({"schema": {"amount": "amt", "step": "amt"}},
            "two canonical fields sharing one source column is refused", "amt")
    rejects({"schema": {"amount": None}},
            "a required field mapped to null is refused", "amount")


def main() -> int:
    print("=" * 70)
    print("SCHEMA -- the data boundary")
    print("=" * 70)
    a_identity_default()
    a_explicit_mapping()
    a_rejections()

    print("\n" + "=" * 70)
    if failures:
        print(f"{len(failures)} CHECK(S) FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd ml-service && ./venv/bin/python test_schema.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'schema'`

- [ ] **Step 3: Write the minimal implementation**

Create `ml-service/schema.py`:

```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd ml-service && ./venv/bin/python test_schema.py`
Expected: PASS — `ALL CHECKS PASSED`

- [ ] **Step 5: Commit**

```bash
git add ml-service/schema.py ml-service/test_schema.py
git commit -m "Phase 14: canonical field table and column-mapping validation

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: normalize() — rename, require, coerce

**Files:**
- Modify: `ml-service/schema.py`
- Modify: `ml-service/test_schema.py`

**Interfaces:**
- Consumes: `load_mapping`, `SchemaError`, `REQUIRED_CANONICAL`, `NUMERIC_CANONICAL`.
- Produces: `class CapabilityReport` with `mapped: list[str]` and
  `absent: list[str]` (Task 3 adds `render()`, Task 6 adds `groups` and
  `active_feature_count`);
  `normalize(df: pd.DataFrame, mapping: dict[str, str]) -> tuple[pd.DataFrame, CapabilityReport]`.
  **This signature does not change again** — later tasks only add to the report.

- [ ] **Step 1: Write the failing test**

Add to `ml-service/test_schema.py`, and add `import pandas as pd` at the top and
`b_normalise_renames()` plus `b_normalise_requires()` to `main()`:

```python
def _sparkov_like() -> "pd.DataFrame":
    """A frame shaped like a real non-PaySim dataset: no balance columns."""
    return pd.DataFrame({
        "hour": [1, 2, 3],
        "category": ["PAYMENT", "TRANSFER", "PAYMENT"],
        "amt": ["100.50", "250.00", "75.25"],
        "cc_num": ["C1", "C2", "C1"],
        "merchant": ["M1", "M1", "M2"],
        "is_fraud": [0, 1, 0],
    })


SPARKOV_MAP = {
    "schema": {
        "step": "hour", "type": "category", "amount": "amt",
        "nameOrig": "cc_num", "nameDest": "merchant", "isFraud": "is_fraud",
        "oldbalanceOrg": None, "newbalanceOrig": None,
        "oldbalanceDest": None, "newbalanceDest": None,
    }
}


def b_normalise_renames() -> None:
    print("\n4. normalize() renames to canonical names")
    mapping = schema.load_mapping(SPARKOV_MAP)
    out, report = schema.normalize(_sparkov_like(), mapping)
    for field in ["step", "type", "amount", "nameOrig", "nameDest", "isFraud"]:
        check(field in out.columns, f"'{field}' exists after normalisation")
    check("amt" not in out.columns, "the original 'amt' name is gone")
    check(out["amount"].dtype.kind == "f",
          "amount was coerced from string to float",
          f"dtype is {out['amount'].dtype}")
    check(sorted(report.absent) == sorted(schema.OPTIONAL_CANONICAL),
          "all four balance columns are reported absent", str(report.absent))


def b_normalise_requires() -> None:
    print("\n5. A missing REQUIRED column is refused")
    df = _sparkov_like().drop(columns=["amt"])
    mapping = schema.load_mapping(SPARKOV_MAP)
    try:
        schema.normalize(df, mapping)
    except schema.SchemaError as e:
        joined = "; ".join(e.errors)
        check("amt" in joined, "the error names the missing SOURCE column", joined)
        check("amount" in joined, "the error names the canonical field too", joined)
    else:
        check(False, "a missing required column must raise SchemaError")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd ml-service && ./venv/bin/python test_schema.py`
Expected: FAIL — `AttributeError: module 'schema' has no attribute 'normalize'`

- [ ] **Step 3: Write the minimal implementation**

Add to `ml-service/schema.py` (and `import pandas as pd` at the top):

```python
def normalize(df: "pd.DataFrame", mapping: dict[str, str]):
    """
    Rename the user's columns to canonical names, coerce numerics, and report
    which optional fields were not supplied.

    Returns (renamed_frame, absent_canonical_names).

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

    mapped = [f for f in CANONICAL_FIELDS if f not in absent]
    return out, CapabilityReport(mapped=mapped, absent=absent)
```

And the minimal report class it returns — `render()` arrives in Task 3:

```python
class CapabilityReport:
    """What the supplied data can and cannot support. Grows across Tasks 3 and 6."""

    def __init__(self, mapped: list[str], absent: list[str]) -> None:
        self.mapped = mapped
        self.absent = absent
        self.groups: dict[str, bool] = {}          # filled by the resolver, Task 6
        self.active_feature_count: int = 0         # filled by the resolver, Task 6
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd ml-service && ./venv/bin/python test_schema.py`
Expected: PASS — `ALL CHECKS PASSED`

- [ ] **Step 5: Commit**

```bash
git add ml-service/schema.py ml-service/test_schema.py
git commit -m "Phase 14: normalise columns at the data boundary

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: Absent balances, and the capability report

**Files:**
- Modify: `ml-service/schema.py`
- Modify: `ml-service/test_schema.py`

**Interfaces:**
- Consumes: Task 2's `normalize`.
- Produces: `CapabilityReport.render() -> str`; and `normalize()` creating the
  four absent balance columns as `0.0`. The signature is unchanged from Task 2.

- [ ] **Step 1: Write the failing test**

Add to `ml-service/test_schema.py`, register `c_absent_balances()` in `main()`,
and change `b_normalise_renames`'s last check to read `report.absent`:

```python
def c_absent_balances() -> None:
    print("\n6. Unsupplied balances use the 'not recorded' encoding")
    mapping = schema.load_mapping(SPARKOV_MAP)
    out, report = schema.normalize(_sparkov_like(), mapping)

    for field in schema.OPTIONAL_CANONICAL:
        check(field in out.columns, f"'{field}' was created")
        check((out[field] == 0.0).all(),
              f"'{field}' is zero -- PaySim's 'not recorded' convention",
              str(out[field].tolist()))

    check(sorted(report.absent) == sorted(schema.OPTIONAL_CANONICAL),
          "the report lists the four absent balance fields", str(report.absent))
    check("amount" in report.mapped, "the report lists what WAS supplied")

    text = report.render()
    check("not recorded" in text.lower(),
          "the rendered report states the encoding used")
    check("oldbalanceOrg" in text,
          "the rendered report names each absent field")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd ml-service && ./venv/bin/python test_schema.py`
Expected: FAIL — `AttributeError: 'CapabilityReport' object has no attribute 'render'`

- [ ] **Step 3: Write the minimal implementation**

In `ml-service/schema.py`, replace the minimal `CapabilityReport` from Task 2
with the full version, and add the zero-encoding to `normalize`:

```python
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
        self.groups: dict[str, bool] = {}          # filled by the resolver, Task 6
        self.active_feature_count: int = 0         # filled by the resolver, Task 6

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
```

Then insert the zero-encoding into `normalize`, immediately before its
`mapped = ...` line:

```python
    # An unsupplied balance becomes PaySim's own "not recorded" encoding: zero.
    # NEVER a reconciling balance -- see the CapabilityReport docstring.
    for field in OPTIONAL_CANONICAL:
        if field not in out.columns:
            out[field] = 0.0
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd ml-service && ./venv/bin/python test_schema.py`
Expected: PASS — `ALL CHECKS PASSED`

- [ ] **Step 5: Commit**

```bash
git add ml-service/schema.py ml-service/test_schema.py
git commit -m "Phase 14: not-recorded balance encoding and the capability report

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Derive `step` from a timestamp column

**Files:**
- Modify: `ml-service/schema.py`
- Modify: `ml-service/test_schema.py`

**Interfaces:**
- Consumes: Task 3's `normalize`.
- Produces: `normalize` additionally honours `mapping["step"]` pointing at a
  datetime-like column, converting it to an integer hour index starting at 1.

- [ ] **Step 1: Write the failing test**

Add to `ml-service/test_schema.py` and register `d_derived_step()` in `main()`:

```python
def d_derived_step() -> None:
    print("\n7. step can be derived from a timestamp column")
    df = pd.DataFrame({
        "ts": ["2026-01-01 00:30:00", "2026-01-01 02:30:00", "2026-01-02 00:30:00"],
        "category": ["PAYMENT", "TRANSFER", "PAYMENT"],
        "amt": [10.0, 20.0, 30.0],
        "cc_num": ["C1", "C2", "C1"],
        "merchant": ["M1", "M1", "M2"],
    })
    cfg = {"schema": {
        "step": "ts", "type": "category", "amount": "amt",
        "nameOrig": "cc_num", "nameDest": "merchant", "isFraud": None,
        "oldbalanceOrg": None, "newbalanceOrig": None,
        "oldbalanceDest": None, "newbalanceDest": None,
    }}
    out, _ = schema.normalize(df, schema.load_mapping(cfg))

    check(out["step"].dtype.kind in "iu", "step is an integer",
          f"dtype is {out['step'].dtype}")
    steps = out["step"].tolist()
    check(steps[0] == 1, "the earliest row is step 1", str(steps))
    check(steps[1] == 3, "two hours later is step 3", str(steps))
    check(steps[2] == 25, "the next day is step 25", str(steps))
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd ml-service && ./venv/bin/python test_schema.py`
Expected: FAIL — step stays a string of dates, `dtype.kind` is `O`

- [ ] **Step 3: Write the minimal implementation**

Add to `ml-service/schema.py`, and call it from `normalize` immediately after
the `rename` line:

```python
def _coerce_step(out: "pd.DataFrame") -> "pd.DataFrame":
    """
    `step` is an integer hour index -- PaySim's unit, and what the velocity
    windows and the rule layer count in.

    Most real datasets have a timestamp instead. Convert it to hours since the
    earliest row, 1-based so that the first hour is step 1 exactly as PaySim
    numbers it. Leave an already-integer column alone.
    """
    if out["step"].dtype.kind in "iu":
        return out

    as_time = pd.to_datetime(out["step"], errors="coerce")
    if as_time.isna().all():
        raise SchemaError([
            "'step' is neither an integer hour index nor a parseable timestamp. "
            f"Sample value: {out['step'].iloc[0]!r}"
        ])

    hours = (as_time - as_time.min()).dt.total_seconds() // 3600
    out["step"] = (hours + 1).astype("int64")
    return out
```

Call site — inside `normalize`, directly after `out = df.rename(...).copy()`:

```python
    out = _coerce_step(out)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd ml-service && ./venv/bin/python test_schema.py`
Expected: PASS — `ALL CHECKS PASSED`

- [ ] **Step 5: Commit**

```bash
git add ml-service/schema.py ml-service/test_schema.py
git commit -m "Phase 14: derive the hour index from a timestamp column

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: Wire the boundary into `load_raw_data`

**Files:**
- Modify: `config.yaml` (add a `schema:` block)
- Modify: `ml-service/features.py:95-134` (`load_raw_data`)
- Modify: `ml-service/test_schema.py`

**Interfaces:**
- Consumes: `schema.load_mapping`, `schema.normalize`.
- Produces: `load_raw_data(cfg)` still returns a `pd.DataFrame`; it additionally
  stores the report on the module attribute `features.LAST_CAPABILITY_REPORT`
  for Task 8 to read.

- [ ] **Step 1: Write the failing test**

Add to `ml-service/test_schema.py` and register `e_identity_is_paysim()` in `main()`:

```python
def e_identity_is_paysim() -> None:
    print("\n8. The default config is an identity mapping (PaySim unchanged)")
    import features
    cfg = features.load_config()
    mapping = schema.load_mapping(cfg)
    for field in schema.CANONICAL_FIELDS:
        check(mapping[field] == field,
              f"default config maps {field} to itself", mapping.get(field))
    check(hasattr(features, "LAST_CAPABILITY_REPORT"),
          "features exposes LAST_CAPABILITY_REPORT for the training report")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd ml-service && ./venv/bin/python test_schema.py`
Expected: FAIL — `features exposes LAST_CAPABILITY_REPORT`

- [ ] **Step 3: Write the minimal implementation**

Add to `config.yaml`, immediately after the `paths:` block:

```yaml
# ----------------------------------------------------------------------------
# Phase 14 -- COLUMN MAPPING. Which of YOUR columns mean what.
#
# Left as-is, every field maps to itself, which is exactly PaySim -- so a PaySim
# file needs no changes here at all.
#
# To use your own data, put YOUR column name on the right:
#     amount: amt
#     nameOrig: cc_num
#     nameDest: merchant
#
# `step` may be an integer hour index OR a timestamp column; a timestamp is
# converted to hours-since-the-earliest-row automatically.
#
# Set a balance field to `null` if your data does not have it. The system will
# encode it as "not recorded" and TELL YOU what that costs, rather than
# quietly training a weaker model.
# ----------------------------------------------------------------------------
schema:
  step: step
  type: type
  amount: amount
  nameOrig: nameOrig
  nameDest: nameDest
  oldbalanceOrg: oldbalanceOrg
  newbalanceOrig: newbalanceOrig
  oldbalanceDest: oldbalanceDest
  newbalanceDest: newbalanceDest
  isFraud: isFraud
```

In `ml-service/features.py`, add `import schema` at the top, add the module
attribute near `PAYSIM_COLUMNS`, and replace the `PAYSIM_COLUMNS` check block
inside `load_raw_data` (currently lines 117-125):

```python
# Set by load_raw_data so train_model.py can print and persist the report
# without threading a return value through four call sites.
LAST_CAPABILITY_REPORT = None
```

Replace:

```python
    # --- Sanity check: are the columns the ones we expect? ---
    missing = [c for c in PAYSIM_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"The CSV is missing expected PaySim columns: {missing}\n"
            f"Columns actually found: {list(df.columns)}\n"
            "You may have downloaded a different fraud dataset."
        )
```

with:

```python
    # --- The data boundary: rename THEIR columns to OUR canonical names. ---
    # Everything downstream speaks canonical names only and never learns that
    # the file called them something else. See schema.py.
    global LAST_CAPABILITY_REPORT
    mapping = schema.load_mapping(cfg)
    df, LAST_CAPABILITY_REPORT = schema.normalize(df, mapping)
    print(LAST_CAPABILITY_REPORT.render())
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd ml-service && ./venv/bin/python test_schema.py`
Expected: PASS — `ALL CHECKS PASSED`

- [ ] **Step 5: Run the regression gate**

```bash
cd ml-service
export RAZORPAY_WEBHOOK_SECRET="$(cat ../.demo-logs/.webhook-secret)"
./venv/bin/python app.py &            # if not already running
for t in test_api test_upi test_webhook test_explain test_genai; do
  echo "--- $t"; ./venv/bin/python $t.py | tail -2
done
```

Expected: `ALL CHECKS PASSED` from all five.

- [ ] **Step 6: Commit**

```bash
git add config.yaml ml-service/features.py ml-service/test_schema.py
git commit -m "Phase 14: normalise at load_raw_data; identity mapping by default

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: FEATURE_GROUPS and the resolver

**Files:**
- Modify: `ml-service/features.py:665-763` (`MODEL_FEATURES`)
- Modify: `ml-service/test_schema.py`

**Interfaces:**
- Consumes: `schema.CapabilityReport`.
- Produces: `FEATURE_GROUPS: dict[str, list[str]]` with keys
  `"A core transaction"`, `"B balance"`, `"C receiver behaviour"`,
  `"D sender behaviour"`;
  `resolve_features(report, enable_sender_group: bool = False) -> list[str]`;
  **`MODEL_FEATURES` keeps its name** and equals
  `resolve_features(<all groups available>)` — `test_explain.py:27` imports it.

- [ ] **Step 1: Write the failing test**

Add to `ml-service/test_schema.py` and register `f_feature_groups()` in `main()`:

```python
def f_feature_groups() -> None:
    print("\n9. Feature groups resolve from what the data supports")
    import features

    counts = {name: len(cols) for name, cols in features.FEATURE_GROUPS.items()}
    check(counts["A core transaction"] == 9, "group A has 9 features", str(counts))
    check(counts["B balance"] == 11, "group B has 11 features", str(counts))
    check(counts["C receiver behaviour"] == 8, "group C has 8 features", str(counts))
    check(counts["D sender behaviour"] == 7, "group D has 7 features", str(counts))

    # PaySim: everything supplied, but sender history is inert (99.85% of
    # senders appear once), so group D stays off and the list is the old 28.
    full = schema.CapabilityReport(mapped=list(schema.CANONICAL_FIELDS), absent=[])
    resolved = features.resolve_features(full, enable_sender_group=False)
    check(len(resolved) == 28, "PaySim resolves to the original 28 features",
          str(len(resolved)))
    check(resolved == list(features.MODEL_FEATURES),
          "MODEL_FEATURES is exactly the resolver's PaySim output")

    # No balances: group B drops out entirely.
    no_bal = schema.CapabilityReport(
        mapped=[f for f in schema.CANONICAL_FIELDS
                if f not in schema.OPTIONAL_CANONICAL],
        absent=list(schema.OPTIONAL_CANONICAL))
    resolved = features.resolve_features(no_bal, enable_sender_group=False)
    check(len(resolved) == 17, "without balances, 17 features remain",
          str(len(resolved)))
    check(not any(f.startswith("errorBalance") for f in resolved),
          "no balance-derived feature survives")

    # No balances but repeat senders: group D compensates.
    resolved = features.resolve_features(no_bal, enable_sender_group=True)
    check(len(resolved) == 24, "group D brings it back to 24", str(len(resolved)))
    check("user_velocity_count" in resolved, "sender velocity is included")
    check(no_bal.groups["B balance"] is False, "the report records B as off")
    check(no_bal.active_feature_count == 24, "the report records the count")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd ml-service && ./venv/bin/python test_schema.py`
Expected: FAIL — `module 'features' has no attribute 'FEATURE_GROUPS'`

- [ ] **Step 3: Write the minimal implementation**

In `ml-service/features.py`, replace the `MODEL_FEATURES = [...]` literal with:

```python
# =============================================================================
# FEATURE GROUPS -- what the model gets to see, decided by what the data has.
#
# Phase 14. Previously this was one fixed list of 28. It is now four groups,
# each switched on only when the supplied data can support it, so that a file
# without balance columns produces a HONESTLY SMALLER model rather than one
# with eleven silently-constant inputs.
# =============================================================================

FEATURE_GROUPS = {
    "A core transaction": [
        "amount",
        "type_CASH_IN", "type_CASH_OUT", "type_DEBIT",
        "type_PAYMENT", "type_TRANSFER",
        "is_risky_type",
        "hour_of_day", "is_night",
    ],
    "B balance": [
        "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
        "errorBalanceOrig", "errorBalanceDest",
        "absErrorBalanceOrig", "absErrorBalanceDest",
        "orig_account_emptied", "dest_was_empty", "amount_to_balance_ratio",
    ],
    "C receiver behaviour": [
        "dest_txn_count", "dest_mean_amount", "dest_std_amount",
        "dest_has_thin_history", "dest_amount_deviation", "dest_amount_ratio",
        "dest_velocity_count", "is_new_receiver",
    ],
    # Built by add_account_profiles(prefix="user") and deliberately OFF for
    # PaySim: 99.85% of senders appear exactly once there, so these columns are
    # constant or a rescaled copy of `amount`. Worse, user_amount_deviation
    # would mean something DIFFERENT live (where customers do have history)
    # than during training -- a train/serve mismatch that silently corrupts
    # live scores. On data with repeat customers that measurement comes out the
    # other way and the group is genuinely informative.
    "D sender behaviour": [
        "user_txn_count", "user_mean_amount", "user_std_amount",
        "user_has_thin_history", "user_amount_deviation", "user_amount_ratio",
        "user_velocity_count",
    ],
}


def resolve_features(report, enable_sender_group: bool = False) -> list[str]:
    """
    Decide the active feature list from what the data actually supplied.

    Group B needs all four balance columns. Group D needs an explicit decision
    from the caller, because whether sender history means anything is a
    MEASUREMENT of the data, not a property of the schema -- Phase 15's
    calibrate.py makes that measurement. Default off preserves PaySim.

    Also records the outcome on the report, so the reason is printed and stored
    alongside the model rather than living only in someone's memory.
    """
    balances_present = not any(
        f in report.absent for f in schema.OPTIONAL_CANONICAL
    )
    active_groups = {
        "A core transaction": True,
        "B balance": balances_present,
        "C receiver behaviour": True,
        "D sender behaviour": bool(enable_sender_group),
    }

    resolved: list[str] = []
    for name, columns in FEATURE_GROUPS.items():
        if active_groups[name]:
            resolved.extend(columns)

    report.groups = active_groups
    report.active_feature_count = len(resolved)
    return resolved


# The canonical PaySim feature list, kept under its original name because
# test_explain.py and the trained model metadata both refer to it.
MODEL_FEATURES = resolve_features(
    schema.CapabilityReport(mapped=list(schema.CANONICAL_FIELDS), absent=[]),
    enable_sender_group=False,
)
```

Note: the long explanatory comment currently sitting inside the
`MODEL_FEATURES` literal about sender-side exclusion moves into the
`"D sender behaviour"` entry above — do not delete it.

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd ml-service && ./venv/bin/python test_schema.py`
Expected: PASS — `ALL CHECKS PASSED`

- [ ] **Step 5: Run the regression gate**

```bash
cd ml-service
for t in test_api test_upi test_webhook test_explain test_genai; do
  echo "--- $t"; ./venv/bin/python $t.py | tail -2
done
```

Expected: all five `ALL CHECKS PASSED`. `test_explain.py` matters most here —
it asserts `MODEL_FEATURES` is unchanged in content.

- [ ] **Step 6: Commit**

```bash
git add ml-service/features.py ml-service/test_schema.py
git commit -m "Phase 14: four feature groups resolved from data capability

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: Seven Group D phrase builders

**Files:**
- Modify: `ml-service/explain.py:65+` (`PHRASE_BUILDERS`)
- Modify: `ml-service/test_schema.py`

**Interfaces:**
- Consumes: `FEATURE_GROUPS["D sender behaviour"]`.
- Produces: seven new `PHRASE_BUILDERS` entries, one per Group D feature.

**Why this is in Phase 14, not Phase 15:** `test_explain.py:57` asserts
`missing = [f for f in MODEL_FEATURES if f not in PHRASE_BUILDERS]` is empty.
More importantly, a SHAP reason with no sentence behind it is precisely what
this system exists not to produce — the explanation layer is the project.

- [ ] **Step 1: Write the failing test**

Add to `ml-service/test_schema.py` and register `g_group_d_phrases()` in `main()`:

```python
def g_group_d_phrases() -> None:
    print("\n10. Every group D feature can be explained in words")
    import features
    from explain import PHRASE_BUILDERS

    for feat in features.FEATURE_GROUPS["D sender behaviour"]:
        check(feat in PHRASE_BUILDERS, f"'{feat}' has a phrase builder")

    for feat in features.FEATURE_GROUPS["D sender behaviour"]:
        if feat not in PHRASE_BUILDERS:
            continue
        text = PHRASE_BUILDERS[feat](3.0)
        check(isinstance(text, str) and len(text) > 10,
              f"'{feat}' produces a readable sentence", repr(text))
        check(not text.endswith("."),
              f"'{feat}' has no trailing full stop, matching the other 28",
              repr(text))
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd ml-service && ./venv/bin/python test_schema.py`
Expected: FAIL — 7 failures, `'user_txn_count' has a phrase builder` and so on

- [ ] **Step 3: Write the minimal implementation**

Add to `ml-service/explain.py` inside the `PHRASE_BUILDERS` dict, after the
receiver-profile entries:

```python
    # --- Group D: the SENDER's own behaviour (Phase 14) ---------------------
    #
    # Off on PaySim, where 99.85% of senders appear exactly once. Live, and on
    # any dataset with repeat customers, these are among the most intuitive
    # reasons a person can be given -- they compare someone against themselves
    # rather than against a population.
    "user_txn_count": lambda v: (
        "this customer has no earlier transactions on record" if v < 1 else
        f"this customer has {int(v)} earlier transactions on record"
    ),
    "user_mean_amount": lambda v: (
        f"this customer's payments usually run around {_money(v)}"
    ),
    "user_std_amount": lambda v: (
        "this customer's payment amounts are usually very consistent"
        if v < 1 else
        f"this customer's payment amounts typically vary by about {_money(v)}"
    ),
    "user_has_thin_history": lambda v: (
        "this customer has almost no history yet, so their normal is not "
        "established" if v else "this customer has enough history to judge "
        "what is normal for them"
    ),
    "user_amount_deviation": lambda v: (
        f"the amount is {v:.1f} standard deviations above this customer's own "
        f"normal" if v > 0 else
        "the amount is within this customer's own normal range"
    ),
    "user_amount_ratio": lambda v: (
        f"the amount is {v:.1f} times this customer's usual payment"
        if v > 1 else "the amount is no larger than this customer's usual payment"
    ),
    "user_velocity_count": lambda v: (
        "this customer has made no other recent transactions" if v < 1 else
        f"this customer made {int(v)} other transactions in the recent window"
    ),
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd ml-service && ./venv/bin/python test_schema.py`
Expected: PASS — `ALL CHECKS PASSED`

- [ ] **Step 5: Run the regression gate**

```bash
cd ml-service && ./venv/bin/python test_explain.py | tail -2
```

Expected: `ALL CHECKS PASSED` — the extra builders are additive, and
`test_explain.py:57` only requires that every `MODEL_FEATURES` entry HAS one.

- [ ] **Step 6: Commit**

```bash
git add ml-service/explain.py ml-service/test_schema.py
git commit -m "Phase 14: plain-language phrases for the sender-behaviour group

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 8: Surface the capability report, and close the gate

**Files:**
- Modify: `ml-service/app.py:254-273` (`/health`)
- Modify: `ml-service/test_schema.py`
- Modify: `docs/DATASET.md`

**Interfaces:**
- Consumes: `features.LAST_CAPABILITY_REPORT`, `features.FEATURE_GROUPS`.
- Produces: `/health` gains a `"data_capability"` object with keys
  `groups: dict[str, bool]`, `active_feature_count: int`, `absent: list[str]`.

- [ ] **Step 1: Write the failing test**

Add to `ml-service/test_schema.py` and register `h_health_reports()` in `main()`:

```python
def h_health_reports() -> None:
    print("\n11. /health reports what the data supports")
    import requests
    try:
        r = requests.get("http://localhost:5001/health", timeout=3)
    except Exception:
        print("  SKIPPED: no server on port 5001")
        return

    body = r.json()
    check("data_capability" in body, "/health has a data_capability block",
          str(list(body.keys())))
    cap = body.get("data_capability", {})
    check("groups" in cap, "it lists the feature groups")
    check("active_feature_count" in cap, "it states the active feature count")
    check(cap.get("active_feature_count") == body.get("feature_count"),
          "the count agrees with feature_count",
          f"{cap.get('active_feature_count')} vs {body.get('feature_count')}")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd ml-service && ./venv/bin/python test_schema.py`
Expected: FAIL — `/health has a data_capability block`

- [ ] **Step 3: Write the minimal implementation**

In `ml-service/app.py`, inside the `/health` handler's returned dict, add:

```python
        # Phase 14 -- what the training data could actually support. Exposed so
        # nobody has to read the training log to find out that eleven features
        # were unavailable.
        "data_capability": {
            # Derived from the columns the model was ACTUALLY trained on, not
            # from a default -- the serving process never calls load_raw_data, so
            # LAST_CAPABILITY_REPORT is None here. Defaulting these to True would
            # advertise a capability the model does not have, which is the exact
            # silent failure this report exists to prevent.
            "groups": {
                name: all(c in FEATURE_COLUMNS for c in cols)
                for name, cols in features.FEATURE_GROUPS.items()
            },
            "active_feature_count": len(FEATURE_COLUMNS),
            "absent": getattr(features.LAST_CAPABILITY_REPORT, "absent", []),
        },
```

Add `import features` to `app.py` if it is not already imported.

- [ ] **Step 4: Restart the service and run the test**

```bash
cd ml-service
pkill -f "app.py"; sleep 2
export RAZORPAY_WEBHOOK_SECRET="$(cat ../.demo-logs/.webhook-secret)"
nohup ./venv/bin/python app.py > ../.demo-logs/ml-service.log 2>&1 &
sleep 5
./venv/bin/python test_schema.py
```

Expected: PASS — `ALL CHECKS PASSED`

- [ ] **Step 5: Document it**

Append a section to `docs/DATASET.md` titled `## Using your own data`, covering:
the `schema:` block with a worked Sparkov-style example; that `step` accepts a
timestamp; that a `null` balance is encoded as "not recorded" and what that
costs (68.3% of importance, 149/150 → 1/150); and that Group D turns on only
when the data has repeat senders, which Phase 15's `calibrate.py` measures.

- [ ] **Step 6: Run the full regression gate**

```bash
cd ml-service
for t in test_api test_upi test_webhook test_explain test_genai test_schema; do
  printf "%-14s " "$t"; ./venv/bin/python $t.py | tail -1
done
```

Expected: six lines, every one `ALL CHECKS PASSED`.

- [ ] **Step 7: Commit**

```bash
git add ml-service/app.py ml-service/test_schema.py docs/DATASET.md
git commit -m "Phase 14: expose data capability on /health and document BYOD

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Deferred to Phase 15

- `calibrate.py` — thresholds derived from the user's own data with evidence.
- The repeat-sender measurement that decides `enable_sender_group` automatically.
  Until then it is an explicit argument, defaulting off.
- The `calibrated_on:` provenance stamp and its mismatch warning.
- The Sparkov end-to-end run.
- **Persisting the capability report into `model_metadata.json`** (spec §7).
  Deferred deliberately: the write only takes effect during a training run, and
  this phase does not retrain — `features.py main()` is a ~50 second, ~4.5 GB
  pass over 6.3M rows. Until then `/health` derives the same truth from
  `FEATURE_COLUMNS`, which is what the model was actually trained on, so no
  capability is over-claimed in the meantime.

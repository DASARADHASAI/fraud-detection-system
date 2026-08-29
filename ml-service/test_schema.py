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

#!/usr/bin/env python3
"""
Seed the dashboard with ordinary traffic before a demonstration.

An empty decision feed reads as an unfinished project. This posts a handful of
unremarkable payments so the feed, the "where payments ended up" chart and the
totals are already alive when the panel first sees the screen.

Every payment here is deliberately BORING. The drama comes from the scenario
buttons; this is the background against which that drama registers. If one of
these ever stops approving, that is a real signal worth investigating -- not
something to paper over by softening the payment.

    ./venv/bin/python demo_seed.py

Run it after both services are up. demo.sh does this for you.
"""

import sys
import requests

BACKEND = "http://localhost:8080/api"

# PaySim's clock: one step = one hour, so step % 24 is the hour of day.
# Pinned for the same reason acceptance_tests.py pins it -- hour_of_day and
# is_night are real model features, so leaving this to the wall clock would
# make the seeded feed look different depending on when it was run.
STEP_AFTERNOON = 254

# Small everyday payments: groceries, fuel, a phone bill, rent to a known payee.
# Balances are arithmetically consistent, because the model reads the residual.
PAYMENTS = [
    ("PAYMENT",    1250.00, "C1001", "M-GROCER",    48000.0,  12000.0, "Mumbai"),
    ("PAYMENT",     680.50, "C1002", "M-FUEL",      31000.0,   8000.0, "Pune"),
    ("PAYMENT",    2199.00, "C1003", "M-PHONE",     57500.0,  22000.0, "Delhi"),
    ("TRANSFER",  18000.00, "C1004", "C-LANDLORD",  92000.0,  45000.0, "Bengaluru"),
    ("PAYMENT",     435.00, "C1005", "M-COFFEE",    14200.0,   6000.0, "Hyderabad"),
    ("PAYMENT",    3120.75, "C1006", "M-PHARMACY",  26800.0,   9500.0, "Chennai"),
    ("CASH_IN",   25000.00, "C1007", "M-SALARY",     4300.0,       0.0, "Kolkata"),
    ("PAYMENT",    1875.00, "C1008", "M-CLOTHING",  38900.0,  17000.0, "Vijayawada"),
    ("PAYMENT",     920.00, "C1009", "M-BOOKS",     21400.0,   5200.0, "Mumbai"),
    ("TRANSFER",   7500.00, "C1010", "C-FAMILY",    64000.0,  31000.0, "Pune"),
    ("PAYMENT",    5400.00, "C1011", "M-ELECTRIC",  47300.0,  19800.0, "Delhi"),
    ("PAYMENT",    1640.25, "C1012", "M-GROCER",    33100.0,  13400.0, "Bengaluru"),
    ("DEBIT",       299.00, "C1013", "M-STREAMING", 18700.0,   7100.0, "Chennai"),
    ("PAYMENT",    2750.00, "C1014", "M-RESTAURANT",41500.0,  15600.0, "Hyderabad"),
    ("TRANSFER",  12000.00, "C1015", "C-SAVINGS",   78000.0,  52000.0, "Mumbai"),
]

CITIES = {
    "Mumbai": (19.0760, 72.8777), "Delhi": (28.7041, 77.1025),
    "Bengaluru": (12.9716, 77.5946), "Hyderabad": (17.3850, 78.4867),
    "Chennai": (13.0827, 80.2707), "Kolkata": (22.5726, 88.3639),
    "Vijayawada": (16.5062, 80.6480), "Pune": (18.5204, 73.8567),
}


def main() -> int:
    try:
        requests.get(f"{BACKEND}/health", timeout=5).raise_for_status()
    except requests.RequestException as exc:
        print(f"  backend not reachable at {BACKEND}: {exc}", file=sys.stderr)
        return 1

    approved = other = 0
    for kind, amount, orig, dest, old_org, old_dest, city in PAYMENTS:
        lat, lon = CITIES[city]
        body = {
            "step": STEP_AFTERNOON,
            "type": kind,
            "amount": amount,
            "nameOrig": orig,
            "nameDest": dest,
            "oldbalanceOrg": old_org,
            "newbalanceOrig": max(old_org - amount, 0.0),
            "oldbalanceDest": old_dest,
            "newbalanceDest": old_dest + amount,
            "trustedDevice": True,
            "location": city,
            "locationLat": lat,
            "locationLon": lon,
        }
        try:
            response = requests.post(f"{BACKEND}/transactions", json=body, timeout=30)
            response.raise_for_status()
        except requests.RequestException as exc:
            print(f"  {orig} -> {dest} failed: {exc}", file=sys.stderr)
            return 1

        decision = response.json().get("decision") or response.json().get("finalDecision", "?")
        if decision == "APPROVE":
            approved += 1
        else:
            other += 1
            # Not fatal, but say so plainly. A seeded payment that does not
            # approve will sit in the feed looking like a fraud alert.
            print(f"  note: {orig} -> {dest} came back {decision}, not APPROVE")

    print(f"seeded {len(PAYMENTS)} payments -- {approved} approved, {other} other")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

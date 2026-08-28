"""
Tests for UniPay's real state -- the ledger, the payee list and device trust.

These replace three JavaScript literals that used to sit in pay.html: a balance,
a contact list and a "known device" switch. Each of them asserted something the
bank is supposed to look up, so each of them now has to be proved.

The most important test here is FORGERY (section 5). Two of the model's four
strongest features are computed from the sender's balance, so if a payer can
state its own balance it chooses the evidence used to judge it.

Needs BOTH services running:
    ./venv/bin/python app.py             # in one terminal
    cd ../backend && mvn spring-boot:run # in another
    ./venv/bin/python test_ledger.py
"""

from __future__ import annotations

import sys
import uuid

import requests


failures: list[str] = []
API = "http://localhost:8080/api"


def check(condition: bool, description: str, detail: str = "") -> None:
    if condition:
        print(f"  PASS  {description}")
    else:
        print(f"  FAIL  {description}")
        if detail:
            print(f"        {detail}")
        failures.append(description)


def balance() -> float:
    return requests.get(f"{API}/account/me", timeout=10).json()["balance"]


def balance_of(handle: str) -> float:
    return requests.get(f"{API}/account/{handle}", timeout=10).json()["balance"]


def me() -> str:
    return requests.get(f"{API}/account/me", timeout=10).json()["handle"]


def payees(handle: str) -> list[dict]:
    return requests.get(f"{API}/account/{handle}/payees", timeout=10).json()


def see_device(fingerprint: str, account: str) -> dict:
    return requests.post(f"{API}/device/seen", timeout=10,
                         json={"fingerprint": fingerprint, "account": account}).json()


# >>> WHY EACH SECTION GETS ITS OWN ACCOUNT <<<
# The burst rule blocks a sender that makes more than 3 payments inside one
# hour, and Java measures that window against the WALL CLOCK over durable H2
# rows (RoutingService.applyLocalRules). So a suite that ran seven payments from
# one account would trip a guard that is working exactly as designed, and would
# fail differently depending on how recently it last ran.
#
# Rather than weaken the rule or juggle `step`, each section that spends money
# opens its own account first. That also exercises a real path: an account
# enters the ledger by being PAID, and is credited by settlement.


def pay(orig: str, dest: str, amount: float, *, claimed_balance: float | None = None,
        txn_type: str = "TRANSFER", trusted: bool = True, step: int = 300) -> dict:
    """Post a payment. `claimed_balance` is what the CALLER asserts it holds."""
    claimed = claimed_balance if claimed_balance is not None else amount * 10
    body = {
        "step": step, "type": txn_type, "amount": amount,
        "nameOrig": orig, "oldbalanceOrg": claimed,
        "newbalanceOrig": max(claimed - amount, 0),
        "nameDest": dest, "oldbalanceDest": 0, "newbalanceDest": amount,
        "trustedDevice": trusted,
    }
    return requests.post(f"{API}/transactions", json=body, timeout=30).json()


def stored(txn_id: int) -> dict:
    body = requests.get(f"{API}/transactions/{txn_id}", timeout=10).json()
    return body.get("transaction", body)


def fresh_wallet(funded: float = 50_000.0) -> str:
    """
    Open a brand-new ledger account by paying money into it.

    There is deliberately no "create an account" endpoint - a customer's app
    must not be able to open accounts - so this uses the only real route: an
    account the bank does not hold pays one it will. Settlement creates the
    receiver and credits it, which is exactly the production path.
    """
    funder = f"funder-{uuid.uuid4().hex[:8]}@otherbank"
    wallet = f"wallet-{uuid.uuid4().hex[:8]}@unipay"
    r = pay(funder, wallet, funded, claimed_balance=funded * 10)
    if r.get("decision") != "APPROVE":
        raise AssertionError(f"could not fund a test wallet: {r}")
    return wallet


# =============================================================================
# 1 -- The account exists, and it is the bank's
# =============================================================================

def test_account() -> None:
    print("\n1. The customer's account")
    r = requests.get(f"{API}/account/me", timeout=10)
    check(r.status_code == 200, "GET /account/me answers")
    a = r.json()
    check("@" in a["handle"], "the account is identified by a UPI address", a["handle"])
    check(isinstance(a["balance"], (int, float)),
          "it carries a real numeric balance", str(a["balance"]))
    check(a["openedAt"] is not None, "it records when it was opened")

    r = requests.get(f"{API}/account/nobody-{uuid.uuid4().hex[:8]}@nowhere", timeout=10)
    check(r.status_code == 404, "an account the bank does not hold is a 404",
          f"got {r.status_code}")


# =============================================================================
# 2 -- Payees are earned, not declared
# =============================================================================

def test_payees_are_earned() -> None:
    print("\n2. The payee list comes from payments that actually completed")
    stranger = f"stranger-{uuid.uuid4().hex[:8]}@oksbi"
    check(payees(stranger) == [],
          "an account that has never paid anyone has no payees")

    mine = fresh_wallet()
    dest = f"payee-{uuid.uuid4().hex[:8]}@ybl"
    before = [p["handle"] for p in payees(mine)]
    check(dest not in before, "the new payee is not on the list beforehand")

    r = pay(mine, dest, 500.0)
    check(r["decision"] == "APPROVE",
          "a small payment to a new UPI ID is approved", str(r.get("decision")))

    after = {p["handle"]: p for p in payees(mine)}
    check(dest in after, "the payee appears only after a payment completed")
    if dest in after:
        check(after[dest]["payments"] == 1, "and it is recorded as paid once",
              str(after[dest]["payments"]))
        check(after[dest]["displayName"].lower().startswith("payee"),
              "a readable name is derived from the UPI address",
              after[dest]["displayName"])

    pay(mine, dest, 250.0)
    again = {p["handle"]: p for p in payees(mine)}
    check(again[dest]["payments"] == 2, "a second payment increments the count",
          str(again[dest]["payments"]))


# =============================================================================
# 3 -- Device trust is earned by having been seen
# =============================================================================

def test_device_trust() -> None:
    print("\n3. Device trust")
    mine = me()
    fp = f"dev-{uuid.uuid4().hex}"

    first = see_device(fp, mine)
    check(first["trusted"] is False,
          "a device the bank has never seen is NOT trusted on its first sighting")
    check(first["seenCount"] == 1, "the first sighting is counted as one")

    second = see_device(fp, mine)
    check(second["trusted"] is True, "the same device is trusted the second time")
    check(second["seenCount"] == 2, "and the sighting count advances")
    check(second["firstSeen"] == first["firstSeen"],
          "firstSeen does not move when the device returns")

    other = see_device(f"dev-{uuid.uuid4().hex}", mine)
    check(other["trusted"] is False,
          "a different device on the same account starts untrusted again")

    r = requests.post(f"{API}/device/seen", json={"account": mine}, timeout=10)
    check(r.status_code == 400, "a sighting with no fingerprint is refused",
          f"got {r.status_code}")


# =============================================================================
# 4 -- The money actually moves, and only when it should
# =============================================================================

def test_settlement() -> None:
    print("\n4. Settlement")
    mine = fresh_wallet()
    dest = f"settle-{uuid.uuid4().hex[:8]}@ybl"

    before = balance_of(mine)
    r = pay(mine, dest, 1000.0)
    check(r["decision"] == "APPROVE", "the control payment is approved",
          str(r.get("decision")))
    after = balance_of(mine)
    check(abs((before - after) - 1000.0) < 0.01,
          "an approved payment debits exactly the amount",
          f"{before} -> {after}")

    credited = requests.get(f"{API}/account/{dest}", timeout=10).json()
    check(abs(credited.get("balance", -1) - 1000.0) < 0.01,
          "and credits the receiver by the same amount", str(credited))

    # A blocked payment must not move a rupee.
    before = balance_of(mine)
    drain = pay(mine, f"mule-{uuid.uuid4().hex[:8]}@ybl", before, trusted=False)
    after = balance_of(mine)
    check(drain["decision"] != "APPROVE",
          "draining the whole balance to a stranger is not approved",
          str(drain.get("decision")))
    check(abs(before - after) < 0.01,
          "a payment that was not approved leaves the balance untouched",
          f"{before} -> {after}")


# =============================================================================
# 5 -- THE IMPORTANT ONE. A payer cannot state its own balance.
# =============================================================================

def test_balance_cannot_be_forged() -> None:
    print("\n5. The balance the model is judged on comes from the ledger")
    mine = fresh_wallet()
    real = balance_of(mine)

    # The attack: claim to be enormously wealthy, so that emptying the account
    # looks like a trivial fraction of it. amount_to_balance_ratio and
    # orig_account_emptied are the model's 3rd and 4th strongest features.
    forged = pay(mine, f"mule-{uuid.uuid4().hex[:8]}@ybl", real,
                 claimed_balance=99_999_999.0, trusted=False)

    row = stored(forged["id"])
    check(abs(row["oldbalanceOrg"] - real) < 0.01,
          "the claimed balance is discarded; the ledger's is stored",
          f"claimed 99,999,999 | ledger {real} | stored {row['oldbalanceOrg']}")
    check(abs(row["newbalanceOrig"]) < 0.01,
          "so the payment is correctly seen as emptying the account",
          str(row["newbalanceOrig"]))
    check(forged["decision"] != "APPROVE",
          "and the forgery does not buy an approval", str(forged.get("decision")))

    check(abs(balance_of(mine) - real) < 0.01,
          "the forged attempt moved no money")


# =============================================================================
# 6 -- Accounts the bank does not hold are left exactly as sent
# =============================================================================

def test_foreign_accounts_untouched() -> None:
    print("\n6. The override applies only to accounts this bank holds")
    # Every existing test suite posts invented account ids. If the ledger
    # rewrote those too, this change would silently alter their meaning.
    orig = f"foreign-{uuid.uuid4().hex[:8]}@otherbank"
    r = pay(orig, f"dest-{uuid.uuid4().hex[:8]}@otherbank", 700.0,
            claimed_balance=250_000.0)
    row = stored(r["id"])
    check(abs(row["oldbalanceOrg"] - 250_000.0) < 0.01,
          "a sender the bank does not hold keeps the balance it declared",
          str(row["oldbalanceOrg"]))
    check(abs(row["newbalanceOrig"] - 249_300.0) < 0.01,
          "and its post-balance is untouched too", str(row["newbalanceOrig"]))


# =============================================================================
# 7 -- The customer's app still cannot learn why
# =============================================================================

def test_customer_side_reveals_nothing() -> None:
    print("\n7. Nothing on the customer's endpoints leaks a reason")
    mine = me()
    forbidden = {"fraudScore", "score", "ruleApplied", "shapReasons",
                 "modelDecision", "reasons", "threshold"}

    account = requests.get(f"{API}/account/me", timeout=10).json()
    check(not (forbidden & set(account)),
          "GET /account/me exposes no score, rule or threshold",
          str(forbidden & set(account)))

    rows = payees(mine)
    leaked = set()
    for row in rows:
        leaked |= forbidden & set(row)
    check(not leaked, "the payee list exposes no score, rule or threshold", str(leaked))

    device = see_device(f"dev-{uuid.uuid4().hex}", mine)
    check(not (forbidden & set(device)),
          "the device check exposes no score, rule or threshold",
          str(forbidden & set(device)))


def services_up() -> bool:
    try:
        return requests.get(f"{API}/health", timeout=3).status_code == 200
    except requests.RequestException:
        return False


def main() -> int:
    print("=" * 70)
    print("LEDGER, PAYEES AND DEVICE TRUST")
    print("=" * 70)
    if not services_up():
        print(f"\n  No backend at {API}")
        print("  start it:  cd ../backend && mvn spring-boot:run")
        return 1

    test_account()
    test_payees_are_earned()
    test_device_trust()
    test_settlement()
    test_balance_cannot_be_forged()
    test_foreign_accounts_untouched()
    test_customer_side_reveals_nothing()

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

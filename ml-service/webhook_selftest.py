"""
Prove the webhook path works BEFORE involving Razorpay.

>>> WHY THIS EXISTS <<<
When a gateway webhook does not arrive, there are four different things that
could be wrong, and they all look identical from the dashboard:

    1. the tunnel is not running, or its URL changed
    2. the URL in the Razorpay dashboard is wrong
    3. the webhook secret does not match
    4. the receiver itself is broken

Debugging all four at once, through someone else's retry schedule, is miserable.
This script settles 3 and 4 in one second against localhost, and settles 1 and 2
by posting through the tunnel URL. Whatever is left is genuinely Razorpay's
configuration, and you can go and look at it knowing that.

Usage:
    ./venv/bin/python webhook_selftest.py
    ./venv/bin/python webhook_selftest.py https://something.trycloudflare.com

The secret comes from RAZORPAY_WEBHOOK_SECRET and must be the SAME value the
service was started with.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import sys
import uuid

import requests


DEFAULT_TARGET = "http://localhost:5001"
SECRET = os.environ.get("RAZORPAY_WEBHOOK_SECRET", "")


def event(amount_paise: int = 4_850_000, payer: str = "selftest@oksbi") -> dict:
    """A payment.authorized event in Razorpay's real shape."""
    return {
        "entity": "event",
        "account_id": "acc_SELFTEST",
        "event": "payment.authorized",
        "contains": ["payment"],
        "payload": {"payment": {"entity": {
            "id": "pay_" + uuid.uuid4().hex[:14],
            "entity": "payment",
            "amount": amount_paise,
            "currency": "INR",
            "status": "authorized",
            "order_id": "order_" + uuid.uuid4().hex[:14],
            "method": "upi",
            "vpa": payer,
            "captured": False,
        }}},
        "created_at": 1567692677,
    }


def post(target: str, body: dict, secret: str) -> requests.Response:
    """
    Sign and post exactly the way Razorpay does.

    The signature covers the RAW bytes we are about to send -- not a
    re-serialisation of the same object. Getting this wrong locally is the
    single most common reason a webhook 401s in production.
    """
    raw = json.dumps(body).encode()
    signature = hmac.new(secret.encode(), raw, hashlib.sha256).hexdigest()
    return requests.post(
        f"{target.rstrip('/')}/webhook/razorpay",
        data=raw,
        headers={"Content-Type": "application/json",
                 "X-Razorpay-Signature": signature},
        timeout=30,
    )


def main() -> int:
    target = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TARGET
    through_tunnel = not target.startswith("http://localhost")

    print("=" * 70)
    print(f"WEBHOOK SELF-TEST  ->  {target}")
    print("=" * 70)

    if not SECRET:
        print("\n  RAZORPAY_WEBHOOK_SECRET is not set in this shell.")
        print("  Export the SAME value the service was started with:")
        print("      export RAZORPAY_WEBHOOK_SECRET='whsec_...'")
        return 1

    # --- 1. Is anything listening at all? ---
    print("\n1. Reachability")
    try:
        r = requests.get(f"{target.rstrip('/')}/webhook/razorpay/recent", timeout=20)
    except requests.RequestException as e:
        print(f"  FAIL  cannot reach {target}")
        print(f"        {type(e).__name__}: {e}")
        if through_tunnel:
            print("\n  The tunnel is probably not running, or its URL has changed.")
            print("  Quick tunnels get a NEW url every restart -- re-copy it into")
            print("  the Razorpay dashboard each time.")
        else:
            print("\n  Start the service:")
            print("      RAZORPAY_WEBHOOK_SECRET=$RAZORPAY_WEBHOOK_SECRET "
                  "./venv/bin/python app.py")
        return 1
    print(f"  PASS  reachable ({r.status_code})")

    status = r.json()
    if not status.get("configured"):
        print("  FAIL  the SERVICE has no webhook secret configured.")
        print("        It was started without RAZORPAY_WEBHOOK_SECRET, so it")
        print("        refuses every webhook. Restart it with the secret set.")
        return 1
    print(f"  PASS  the service has a secret configured")
    print(f"        merchant VPA: {status['merchant_vpa']}")
    print(f"        events received so far: {status['received']}")

    # --- 2. A correctly signed event ---
    print("\n2. A correctly signed payment.authorized")
    r = post(target, event(), SECRET)
    if r.status_code == 401:
        print("  FAIL  401 -- the secret in THIS shell does not match the one")
        print("        the service was started with. They must be identical.")
        return 1
    if r.status_code != 200:
        print(f"  FAIL  {r.status_code}: {r.text[:300]}")
        return 1

    body = r.json()
    print(f"  PASS  accepted and scored")
    print(f"        decision     {body['decision']}  (score {body['fraud_score']})")
    print(f"        enforceable  {body['enforceable']}")
    print(f"        payer        {body['razorpay']['payer_vpa']}")
    print(f"        amount       Rs {body['razorpay']['amount_rupees']:,.2f} "
          f"(from {body['razorpay']['amount_paise']:,} paise)")

    # --- 3. A forged signature must be refused ---
    print("\n3. A forged signature")
    r = post(target, event(), SECRET + "_wrong")
    if r.status_code == 401:
        print("  PASS  refused with 401, as it must be")
    else:
        print(f"  FAIL  a forged signature got {r.status_code}, expected 401.")
        print("        Do NOT expose this endpoint publicly until that is fixed.")
        return 1

    # --- 4. Did the receiver actually record all that? ---
    print("\n4. The receiver's own record")
    r = requests.get(f"{target.rstrip('/')}/webhook/razorpay/recent", timeout=20)
    events = r.json()["events"]
    scored = [e for e in events if e.get("outcome") == "SCORED"]
    rejected = [e for e in events if e.get("outcome") == "REJECTED"]
    print(f"  PASS  {len(scored)} scored, {len(rejected)} rejected, "
          f"{len(events)} total in the log")

    print("\n" + "=" * 70)
    if through_tunnel:
        print("THE PUBLIC PATH WORKS.")
        print("=" * 70)
        print(f"\nPaste this into the Razorpay dashboard as the webhook URL:")
        print(f"\n    {target.rstrip('/')}/webhook/razorpay\n")
        print("Subscribe to:  payment.authorized")
        print("Use the SAME secret you exported here.")
        print("\nThen make a test payment and watch it arrive:")
        print("    curl -s localhost:5001/webhook/razorpay/recent | python3 -m json.tool")
    else:
        print("THE LOCAL PATH WORKS.")
        print("=" * 70)
        print("\nNext: start a tunnel so Razorpay can reach you, then re-run")
        print("this script against the public URL it prints:")
        print("\n    ./razorpay-live.sh\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

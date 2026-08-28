# Going live on Razorpay

*Phase 13. Puts the fraud engine behind a public URL so a real payment gateway
can reach it, and proves the path works before Razorpay is involved.*

The engine itself does not change. This is plumbing: a public address, a shared
secret, and a way to see what arrived.

---

## What you do, and what the scripts do

| | |
|---|---|
| **You** | create the Razorpay account, choose the webhook secret, paste the URL into their dashboard, make the test payment |
| **The scripts** | preflight every local thing that can be wrong, raise a tunnel, verify the public path end to end, and show you what landed |

Nothing here handles your API keys. The webhook secret is a value **you invent**
and export into your shell — it is never issued to you, and it never goes in the
repository.

---

## The five-minute version

```bash
export RAZORPAY_WEBHOOK_SECRET='whsec_pick_something_long'

# terminal 1
cd ml-service
RAZORPAY_WEBHOOK_SECRET="$RAZORPAY_WEBHOOK_SECRET" ./venv/bin/python app.py

# terminal 2
brew install cloudflared          # once
./razorpay-live.sh
```

`razorpay-live.sh` refuses to raise a tunnel until the local path is proved, then
prints the exact URL and event to configure. Paste them into
**Dashboard → Settings → Webhooks**, in **test mode**, subscribed to
`payment.authorized`.

Then make a test payment and watch it arrive:

```bash
curl -s localhost:5001/webhook/razorpay/recent | python3 -m json.tool
```

---

## Why a tunnel

Razorpay's servers cannot reach `localhost`. A quick tunnel gives your laptop a
temporary public address that forwards to port 5001. Nothing is deployed
anywhere; the address dies when you stop the tunnel.

`cloudflared` is the choice here because a quick tunnel needs **no account** —
ngrok now requires a signup. The cost is that the URL changes on every restart,
so re-paste it into the dashboard each time. Say that plainly if a panel asks
why the URL in your screenshot differs from the one on screen.

---

## The four things that go wrong, and how to tell them apart

When a webhook does not arrive, four different faults look identical from the
Razorpay dashboard. Debugging them simultaneously, through someone else's retry
schedule, is miserable — so `webhook_selftest.py` settles them one at a time.

| Symptom | Cause | Fix |
|---|---|---|
| Self-test can't reach the target | tunnel not running, or its URL changed | restart `./razorpay-live.sh`, re-paste the new URL |
| `503` / "no webhook secret configured" | the **service** was started without the secret | restart `app.py` with `RAZORPAY_WEBHOOK_SECRET` set |
| `401` on a correctly signed event | the secret in your **shell** differs from the service's | export the same value in both |
| Everything passes, nothing arrives from Razorpay | the dashboard URL or the event subscription is wrong | check for `/webhook/razorpay` on the end, and `payment.authorized` ticked |

```bash
./razorpay-live.sh --check                       # local only, no tunnel
./ml-service/venv/bin/python ml-service/webhook_selftest.py https://xyz.trycloudflare.com
```

`GET /webhook/razorpay/recent` is the arbiter. A rejected entry with
`signature_present: true` means the secret is wrong. **Nothing at all** means
Razorpay never reached you, which is a URL problem, not a signature problem.
Those are different faults and this is how you separate them.

---

## `payment.authorized`, not `payment.captured`

Subscribe to `payment.authorized`. It is the only event at which a decision is
still possible: the funds are held, nobody has taken them, and with **manual
capture** enabled the merchant still chooses capture or void.

`payment.captured` is scored too, but the response comes back
`enforceable: false` with a note, because the money has already moved. A verdict
there can only prompt a refund.

To make the route genuinely enforceable, turn **auto-capture off** in Razorpay
so that authorisation and capture are separate steps. Otherwise you are
observing, not deciding — and you should say so rather than let it be assumed.

---

## What this proves, and what it does not

**Proves:** the engine accepts real, signature-verified traffic from a real
payment company, in their payload shape, and answers inside the authorisation
window. The integration is genuine.

**Does not prove:** that the engine detects merchant fraud. Two measured reasons,
both in [INTEGRATION.md](INTEGRATION.md) §7a:

- A merchant cannot see the **payer's** balance, where 68.3% of the model's
  importance lives. Measured: fraud blocking falls from 149/150 to **1/150**,
  and **38% of fraud is silently approved**.
- **PaySim contains no merchant fraud at all** — all 8,213 frauds are TRANSFER or
  CASH_OUT, and `PAYMENT` has zero. There is nothing in the data to detect.

This is the honest framing, and it is a stronger position than pretending
otherwise: *we built the merchant route, measured it, and proved with numbers
why fraud detection belongs at the bank.*

---

## Security notes

- The secret lives in the environment, never in the repository. If it leaks,
  rotate it in the dashboard and restart the service.
- With no secret configured the endpoint returns **503 to everything** rather
  than running unauthenticated. An unverified webhook endpoint is an open door.
- `/webhook/razorpay/recent` holds payer UPI addresses in memory. It is a
  localhost diagnostic — it is bounded to 50 entries, never persisted, and
  should not be exposed through the tunnel in any real deployment.
- Stop the tunnel when you are done: `./razorpay-live.sh --stop`. While it is
  up, your port 5001 is on the public internet.

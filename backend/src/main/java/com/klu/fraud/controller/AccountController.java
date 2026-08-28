package com.klu.fraud.controller;

import com.klu.fraud.model.Account;
import com.klu.fraud.service.LedgerService;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * What a customer's payment app is allowed to ask the bank.
 *
 * <pre>
 *   GET  /api/account/me              the customer's real balance
 *   GET  /api/account/{h}/payees      who they have actually paid before
 *   POST /api/device/seen             has this bank seen this device before
 * </pre>
 *
 * <p>>>> WHAT IS DELIBERATELY ABSENT <<<
 * There is no endpoint here that returns a score, a threshold, a rule or a
 * reason. UniPay is the customer's side, and the customer's side must not be
 * able to learn why a payment was stopped - a fraudster who learns which signal
 * caught them learns how to avoid it next time. Everything on this controller
 * is account state the customer already owns and could read from a passbook.
 */
@RestController
@RequestMapping("/api")
@CrossOrigin
public class AccountController {

    private final LedgerService ledger;

    public AccountController(LedgerService ledger) {
        this.ledger = ledger;
    }

    /**
     * The customer's own account. Opened on first call, so a fresh database
     * yields a genuinely new customer rather than a pre-populated one.
     */
    @GetMapping("/account/me")
    public Map<String, Object> me() {
        return view(ledger.customerAccount());
    }

    @GetMapping("/account/{handle}")
    public ResponseEntity<Map<String, Object>> account(@PathVariable String handle) {
        return ledger.find(handle)
                .map(a -> ResponseEntity.ok(view(a)))
                .orElseGet(() -> ResponseEntity.status(HttpStatus.NOT_FOUND)
                        .body(Map.of("error", "no such account", "handle", handle)));
    }

    /**
     * The payee list. Empty on a fresh account, and that is the correct answer:
     * the first payment to anybody is a payment to a new receiver, which the
     * quick-signal layer reads as a genuine signal.
     */
    @GetMapping("/account/{handle}/payees")
    public List<Map<String, Object>> payees(@PathVariable String handle) {
        return ledger.payeesOf(handle);
    }

    /**
     * Register a device sighting and report whether it was known beforehand.
     *
     * <p>Trust is earned here, not asserted. A browser that has never been seen
     * comes back {@code trusted: false} on its first payment, every time.
     */
    @PostMapping("/device/seen")
    public ResponseEntity<Map<String, Object>> deviceSeen(
            @RequestBody Map<String, String> body) {

        String fingerprint = body.get("fingerprint");
        if (fingerprint == null || fingerprint.isBlank()) {
            return ResponseEntity.badRequest()
                    .body(Map.of("error", "fingerprint is required"));
        }
        String account = body.getOrDefault("account", ledger.customerHandle());
        return ResponseEntity.ok(ledger.seeDevice(fingerprint, account));
    }

    private Map<String, Object> view(Account a) {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("handle", a.getHandle());
        out.put("displayName", a.getDisplayName());
        out.put("balance", a.getBalance());
        out.put("merchant", a.getMerchant());
        out.put("openedAt", a.getOpenedAt());
        return out;
    }
}

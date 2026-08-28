package com.klu.fraud.service;

import com.klu.fraud.model.Account;
import com.klu.fraud.model.KnownDevice;
import com.klu.fraud.model.Transaction;
import com.klu.fraud.repository.AccountRepository;
import com.klu.fraud.repository.KnownDeviceRepository;
import com.klu.fraud.repository.TransactionRepository;
import java.time.Instant;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

/**
 * The bank's own record of who holds what, which devices it has seen, and who
 * has genuinely been paid before.
 *
 * <p>>>> WHAT THIS CLASS REPLACED <<<
 * Three JavaScript literals in {@code pay.html}: a balance, a contact list, and
 * a "known device" switch. Each of them asserted something the bank is supposed
 * to <i>look up</i>. Moving them here changes three claims from "trust our
 * page" into "the bank checked":
 *
 * <ul>
 *   <li><b>Balance</b> - read from the ledger, never from the request. A
 *       client-asserted balance is forgeable, and two of the model's four
 *       strongest features are computed from it.</li>
 *   <li><b>Payees</b> - derived from payments that actually completed, so
 *       "established payee" is earned rather than declared.</li>
 *   <li><b>Device trust</b> - earned by having been seen before. A brand-new
 *       browser is untrusted on its first payment, every time.</li>
 * </ul>
 *
 * <p>>>> WHAT IS STILL NOT REAL, STATED PLAINLY <<<
 * The opening balance has to come from somewhere, and no real bank is behind
 * this. {@code unipay.opening-balance} in application.properties is the one
 * invented number, and it is invented once, at account opening, rather than
 * re-asserted on every payment. Everything after that is arithmetic the bank
 * performed on payments that actually happened.
 */
@Service
public class LedgerService {

    private static final Logger log = LoggerFactory.getLogger(LedgerService.class);

    private final AccountRepository accounts;
    private final KnownDeviceRepository devices;
    private final TransactionRepository transactions;

    private final String customerHandle;
    private final double openingBalance;

    public LedgerService(AccountRepository accounts,
                         KnownDeviceRepository devices,
                         TransactionRepository transactions,
                         @Value("${unipay.customer-handle:you@unipay}") String customerHandle,
                         @Value("${unipay.opening-balance:100000}") double openingBalance) {
        this.accounts = accounts;
        this.devices = devices;
        this.transactions = transactions;
        this.customerHandle = customerHandle;
        this.openingBalance = openingBalance;
    }

    public String customerHandle() {
        return customerHandle;
    }

    // ------------------------------------------------------------------------
    // Accounts
    // ------------------------------------------------------------------------

    /**
     * The customer's own account, opened on first use.
     *
     * <p>Opening it lazily rather than seeding it at start-up matters: a fresh
     * database produces a genuinely fresh customer with no history, no payees
     * and no known devices, which is exactly the state a new user is in.
     */
    @Transactional
    public Account customerAccount() {
        return accounts.findById(customerHandle).orElseGet(() -> {
            log.info("Opening account {} with an opening balance of {}",
                    customerHandle, openingBalance);
            return accounts.save(new Account(customerHandle, "You", openingBalance, false));
        });
    }

    public Optional<Account> find(String handle) {
        if (handle == null || handle.isBlank()) {
            return Optional.empty();
        }
        return accounts.findById(handle.trim().toLowerCase());
    }

    /**
     * Record a counterparty the first time it is paid.
     *
     * <p>The bank does not know a stranger's balance - that sits in another
     * bank's core, behind NPCI - so a newly seen payee starts at zero here and
     * only accumulates what it has been paid through this system. See
     * docs/INTEGRATION.md section 6 for why that blindness is real and what it
     * costs.
     */
    @Transactional
    public Account remember(String handle, boolean merchant) {
        String key = handle.trim().toLowerCase();
        return accounts.findById(key).orElseGet(() ->
                accounts.save(new Account(key, displayNameFor(key), 0.0, merchant)));
    }

    /** "ravi@unipay" -> "Ravi", "chai.point@ubi" -> "Chai Point". */
    public static String displayNameFor(String handle) {
        String local = handle.contains("@") ? handle.substring(0, handle.indexOf('@')) : handle;
        String[] parts = local.replace('_', ' ').replace('.', ' ').split("\\s+");
        StringBuilder name = new StringBuilder();
        for (String part : parts) {
            if (part.isBlank()) {
                continue;
            }
            if (name.length() > 0) {
                name.append(' ');
            }
            name.append(Character.toUpperCase(part.charAt(0)))
                .append(part.length() > 1 ? part.substring(1) : "");
        }
        return name.length() == 0 ? handle : name.toString();
    }

    /**
     * Move the money. Called only after a payment has been APPROVED.
     *
     * <p>Deliberately separate from scoring: the ledger changes because a
     * decision was made, never in order to make one.
     */
    @Transactional
    public void settle(Transaction txn) {
        find(txn.getNameOrig()).ifPresent(payer -> {
            payer.setBalance(Math.max(0.0, payer.getBalance() - txn.getAmount()));
            accounts.save(payer);
        });
        find(txn.getNameDest()).ifPresent(payee -> {
            payee.setBalance(payee.getBalance() + txn.getAmount());
            accounts.save(payee);
        });
    }

    // ------------------------------------------------------------------------
    // Payees
    // ------------------------------------------------------------------------

    /** Everyone this account has successfully paid, most recent first. */
    public List<Map<String, Object>> payeesOf(String handle) {
        List<Map<String, Object>> out = new ArrayList<>();
        for (Object[] row : transactions.findPayeesOf(handle)) {
            String payee = (String) row[0];
            Map<String, Object> entry = new LinkedHashMap<>();
            entry.put("handle", payee);
            entry.put("displayName",
                    find(payee).map(Account::getDisplayName).orElse(displayNameFor(payee)));
            entry.put("merchant", find(payee).map(Account::getMerchant).orElse(false));
            entry.put("payments", ((Number) row[1]).longValue());
            entry.put("lastPaidAt", row[2]);
            out.add(entry);
        }
        return out;
    }

    // ------------------------------------------------------------------------
    // Devices
    // ------------------------------------------------------------------------

    /**
     * Register that a device is being used, and say whether the bank had seen
     * it before <i>this moment</i>.
     *
     * <p>The ordering is the whole point. We answer with what we knew BEFORE
     * recording the sighting, so a first-ever visit reports {@code trusted:
     * false} even though the row now exists. Recording first would make every
     * device trusted on arrival, which is the same class of mistake as scoring
     * a transaction after adding it to its own history.
     */
    @Transactional
    public Map<String, Object> seeDevice(String fingerprint, String accountHandle) {
        String key = fingerprint.trim();
        Optional<KnownDevice> existing = devices.findById(key);
        boolean knownBefore = existing.isPresent();

        KnownDevice device = existing.orElseGet(() -> new KnownDevice(key, accountHandle));
        if (knownBefore) {
            device.setLastSeen(Instant.now());
            device.setSeenCount(device.getSeenCount() + 1);
        }
        devices.save(device);

        Map<String, Object> out = new LinkedHashMap<>();
        out.put("fingerprint", key);
        out.put("trusted", knownBefore);
        out.put("firstSeen", device.getFirstSeen());
        out.put("seenCount", device.getSeenCount());
        out.put("note", knownBefore
                ? "This bank has seen this device before."
                : "First time this bank has seen this device - not trusted yet.");
        return out;
    }
}

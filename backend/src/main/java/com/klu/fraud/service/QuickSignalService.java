package com.klu.fraud.service;

import com.klu.fraud.model.Decision;
import com.klu.fraud.model.Transaction;
import com.klu.fraud.repository.TransactionRepository;
import java.util.LinkedHashSet;
import java.util.Set;
import org.springframework.stereotype.Service;

/**
 * Cheap checks that try to settle a borderline payment BEFORE bothering the
 * customer with an OTP.
 *
 * <p>Every OTP is friction. If we can answer "is this fine?" from data the bank
 * already holds, we should, and only ask the customer when we genuinely cannot
 * tell. These checks run in a fixed order, most decisive first.
 *
 * <p>>>> WHY A TRUSTED DEVICE IS NOT ENOUGH ON ITS OWN. <<<
 * The tempting shortcut is "it came from their usual phone, approve it". That
 * is wrong, and knowledge_base/case-fraud-006.json is the reason: in an
 * authorised push payment fraud, the victim is talked into making the payment
 * themselves, from their own device, and would happily confirm an OTP too.
 * A trusted device proves IDENTITY. It says nothing about INTENT. So here it
 * only ever acts as a supporting signal alongside an established payee - never
 * as a clearance by itself.
 */
@Service
public class QuickSignalService {

    /**
     * Accounts already known to be involved in fraud.
     *
     * <p>Hard-coded for the demo. In production this is a shared industry feed
     * updated continuously; the lookup is the same either way.
     */
    private static final Set<String> KNOWN_FRAUD_ACCOUNTS = new LinkedHashSet<>(Set.of(
            "C000000001", "C-KNOWN-FRAUD", "C-MULE-001", "C-BLACKLISTED"
    ));

    private final TransactionRepository transactions;

    public QuickSignalService(TransactionRepository transactions) {
        this.transactions = transactions;
    }

    /** The outcome of the quick checks. */
    public record Outcome(Decision decision, String signal, String detail) {
        public boolean resolved() {
            return signal != null;
        }
    }

    /**
     * Try to settle a borderline payment.
     *
     * @param txn the payment, already saved so it has an id
     * @return an Outcome; {@code resolved()} is false when the checks could not
     *         decide and the customer must be asked
     */
    public Outcome evaluate(Transaction txn) {

        // ---- 1. Is the receiver already known to be a fraudster? ------------
        // Decisive on its own. No score and no customer confirmation should be
        // able to send money to an account we already know is stolen.
        if (KNOWN_FRAUD_ACCOUNTS.contains(txn.getNameDest())) {
            return new Outcome(Decision.BLOCK, "known_fraud_receiver",
                    "The receiving account " + txn.getNameDest()
                    + " is on the known-fraud list, so the payment was stopped "
                    + "without asking the customer.");
        }

        // ---- 2. Has this customer paid this exact receiver before? ----------
        // An established payee that has already gone through successfully is
        // the strongest "this is routine" signal available cheaply.
        long previous = transactions.countPreviousApprovedPayments(
                txn.getNameOrig(), txn.getNameDest(), txn.getId());
        if (previous >= 1) {
            boolean trusted = Boolean.TRUE.equals(txn.getTrustedDevice());
            if (trusted) {
                return new Outcome(Decision.APPROVE, "known_payee_trusted_device",
                        "This customer has paid " + txn.getNameDest() + " "
                        + previous + " time(s) before, and this payment came "
                        + "from a device they have used before.");
            }
            // Established payee but an unfamiliar device: better than nothing,
            // still not enough. A stolen phone pays the victim's usual payees
            // exactly like the victim would.
            return new Outcome(null, null,
                    "Payee is established (" + previous + " previous payments) "
                    + "but the device is unrecognised, so the customer will be asked.");
        }

        // ---- 3. Trusted device alone -> deliberately NOT sufficient ---------
        if (Boolean.TRUE.equals(txn.getTrustedDevice())) {
            return new Outcome(null, null,
                    "The device is recognised, but a trusted device proves who "
                    + "is paying, not whether they meant to. New payee, so the "
                    + "customer will be asked.");
        }

        // ---- Could not settle it. Ask the customer. -------------------------
        return new Outcome(null, null,
                "No quick signal could settle this: a new payee from an "
                + "unrecognised device.");
    }

    public Set<String> getKnownFraudAccounts() {
        return KNOWN_FRAUD_ACCOUNTS;
    }
}

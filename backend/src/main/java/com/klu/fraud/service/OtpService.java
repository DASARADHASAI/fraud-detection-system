package com.klu.fraud.service;

import com.klu.fraud.config.SharedConfig;
import com.klu.fraud.model.Decision;
import com.klu.fraud.model.OtpChallenge;
import com.klu.fraud.model.OtpStatus;
import com.klu.fraud.model.ReviewLog;
import com.klu.fraud.model.Transaction;
import com.klu.fraud.repository.OtpChallengeRepository;
import com.klu.fraud.repository.ReviewLogRepository;
import com.klu.fraud.repository.TransactionRepository;
import java.security.SecureRandom;
import java.time.Instant;
import java.util.List;
import java.util.Optional;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

/**
 * The automatic step-up: asking the customer instead of asking an analyst.
 *
 * <p>This is the piece that lets the system have no human reviewer. When the
 * model is uncertain, a traditional bank queues the payment for a fraud analyst
 * who knows nothing about the customer's life. Here we ask the one person who
 * definitively knows whether the payment is genuine: the customer.
 *
 * <p>>>> SILENCE IS TREATED AS A DENIAL. <<<
 * If the customer does not answer before the deadline, the payment is BLOCKED,
 * not approved. Getting this backwards would hand an attacker a trivial win:
 * make the payment, stay quiet, let the timer run out. The cost of the correct
 * choice is a blocked genuine payment and an annoyed customer; the cost of the
 * wrong one is stolen money. Those are not symmetric.
 *
 * <p>The OTP is SIMULATED - no SMS is sent. The code is returned to the caller
 * so the dashboard can pop it up.
 */
@Service
public class OtpService {

    private static final Logger log = LoggerFactory.getLogger(OtpService.class);
    private static final SecureRandom RANDOM = new SecureRandom();

    private final OtpChallengeRepository challenges;
    private final TransactionRepository transactions;
    private final ReviewLogRepository reviewLog;
    private final SharedConfig config;

    public OtpService(OtpChallengeRepository challenges,
                      TransactionRepository transactions,
                      ReviewLogRepository reviewLog,
                      SharedConfig config) {
        this.challenges = challenges;
        this.transactions = transactions;
        this.reviewLog = reviewLog;
        this.config = config;
    }

    /** Create and "send" a challenge for an uncertain payment. */
    @Transactional
    public OtpChallenge sendChallenge(Transaction txn) {
        OtpChallenge challenge = new OtpChallenge();
        challenge.setTransactionId(txn.getId());
        challenge.setCustomerAccount(txn.getNameOrig());
        challenge.setCode(String.format("%06d", RANDOM.nextInt(1_000_000)));
        challenge.setSentAt(Instant.now());
        challenge.setExpiresAt(Instant.now().plusSeconds(config.getOtpTimeoutSeconds()));
        challenge.setStatus(OtpStatus.PENDING);
        challenge.setQuestion(String.format(
                "Did you just send Rs %,.2f to %s? Reply YES if this was you, "
                + "NO if it was not.",
                txn.getAmount(), txn.getNameDest()));

        challenge = challenges.save(challenge);

        reviewLog.save(new ReviewLog(txn.getId(), "OTP_SENT", "OTP_CONFIRM",
                "Challenge " + challenge.getId() + " sent to " + txn.getNameOrig()
                + ", expires in " + config.getOtpTimeoutSeconds() + "s. "
                + "No reply will be treated as a denial."));

        log.info("OTP {} sent for transaction {} (code {} - simulated, no SMS)",
                challenge.getId(), txn.getId(), challenge.getCode());
        return challenge;
    }

    /**
     * The customer answers.
     *
     * @param confirmed true = "yes that was me", false = "no it was not"
     * @return the final decision for the payment
     */
    @Transactional
    public Decision respond(Long transactionId, boolean confirmed) {
        Optional<OtpChallenge> maybe =
                challenges.findByTransactionIdAndStatus(transactionId, OtpStatus.PENDING);

        if (maybe.isEmpty()) {
            throw new IllegalStateException(
                    "No pending OTP challenge for transaction " + transactionId
                    + " (it may already have been answered or timed out)");
        }

        OtpChallenge challenge = maybe.get();

        // Check expiry at answer time too, not only in the scheduled sweep -
        // a reply that arrives one second after the deadline must not count.
        if (challenge.isExpired()) {
            return expire(challenge, "the reply arrived after the deadline");
        }

        challenge.setStatus(confirmed ? OtpStatus.CONFIRMED : OtpStatus.DENIED);
        challenge.setRespondedAt(Instant.now());
        challenges.save(challenge);

        Decision decision = confirmed ? Decision.APPROVE : Decision.BLOCK;
        applyDecision(transactionId, decision,
                confirmed ? "Customer confirmed the payment by OTP."
                          : "Customer denied the payment by OTP.");

        reviewLog.save(new ReviewLog(transactionId, "OTP_ANSWERED", decision.name(),
                confirmed ? "Customer answered YES - payment approved."
                          : "Customer answered NO - payment blocked."));

        log.info("OTP for transaction {} answered {} -> {}",
                transactionId, confirmed ? "YES" : "NO", decision);
        return decision;
    }

    /**
     * Sweep for challenges nobody answered.
     *
     * <p>Runs every 5 seconds. Without this, an unanswered payment would sit in
     * OTP_CONFIRM forever and never actually be decided - which would quietly
     * reintroduce the human step this system is built to avoid.
     */
    @Scheduled(fixedDelay = 5000)
    @Transactional
    public void expireStaleChallenges() {
        List<OtpChallenge> pending = challenges.findByStatus(OtpStatus.PENDING);
        for (OtpChallenge challenge : pending) {
            if (challenge.isExpired()) {
                expire(challenge, "no reply within " + config.getOtpTimeoutSeconds()
                        + " seconds");
            }
        }
    }

    private Decision expire(OtpChallenge challenge, String why) {
        challenge.setStatus(OtpStatus.TIMED_OUT);
        challenge.setRespondedAt(Instant.now());
        challenges.save(challenge);

        applyDecision(challenge.getTransactionId(), Decision.BLOCK,
                "No response to the OTP challenge - " + why
                + ". Treated as a denial.");

        reviewLog.save(new ReviewLog(challenge.getTransactionId(), "OTP_TIMEOUT",
                "BLOCK", "Blocked automatically: " + why
                + ". Silence is treated as a denial, never as consent."));

        log.info("OTP {} for transaction {} timed out -> BLOCK",
                challenge.getId(), challenge.getTransactionId());
        return Decision.BLOCK;
    }

    private void applyDecision(Long transactionId, Decision decision, String note) {
        transactions.findById(transactionId).ifPresent(txn -> {
            txn.setFinalDecision(decision);
            transactions.save(txn);
        });
    }

    public Optional<OtpChallenge> findPending(Long transactionId) {
        return challenges.findByTransactionIdAndStatus(transactionId, OtpStatus.PENDING);
    }

    public List<OtpChallenge> recent() {
        return challenges.findAllByOrderBySentAtDesc();
    }
}

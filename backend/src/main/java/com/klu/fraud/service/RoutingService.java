package com.klu.fraud.service;

import com.klu.fraud.config.SharedConfig;
import com.klu.fraud.dto.PredictionResult;
import com.klu.fraud.dto.TransactionRequest;
import com.klu.fraud.model.Decision;
import com.klu.fraud.model.OtpChallenge;
import com.klu.fraud.model.ReviewLog;
import com.klu.fraud.model.Transaction;
import com.klu.fraud.repository.ReviewLogRepository;
import com.klu.fraud.repository.TransactionRepository;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

/**
 * THE AUTOMATIC ROUTER. No human step exists anywhere in this class.
 *
 * <p>A payment goes through five stages, in this order:
 *
 * <ol>
 *   <li><b>Store it</b> - so nothing is lost if a later stage fails.</li>
 *   <li><b>Score it</b> - ask Python; fall back locally if it is down.</li>
 *   <li><b>Apply local rules</b> - burst checks Java can answer from H2.</li>
 *   <li><b>Route it</b> - low score approve, high score block, otherwise
 *       step up.</li>
 *   <li><b>Step up if needed</b> - try the quick signals first, and only ask
 *       the customer by OTP when those cannot settle it.</li>
 * </ol>
 *
 * <p>>>> ONE INVARIANT GOVERNS THE WHOLE CLASS: NOTHING MAY SOFTEN A DECISION. <<<
 * Rules, quick signals and the local fallback may all make a payment MORE
 * restricted (APPROVE -> OTP_CONFIRM -> BLOCK). Nothing may move it the other
 * way, with exactly one exception: the customer answering YES to an OTP, which
 * is the one piece of evidence nothing else in the system has access to.
 * {@link Decision#isMoreSevereThan} enforces this at every step.
 */
@Service
public class RoutingService {

    private static final Logger log = LoggerFactory.getLogger(RoutingService.class);

    private final TransactionRepository transactions;
    private final ReviewLogRepository reviewLog;
    private final MlServiceClient mlClient;
    private final QuickSignalService quickSignals;
    private final OtpService otpService;
    private final SharedConfig config;
    private final ObjectMapper json = new ObjectMapper();

    public RoutingService(TransactionRepository transactions,
                          ReviewLogRepository reviewLog,
                          MlServiceClient mlClient,
                          QuickSignalService quickSignals,
                          OtpService otpService,
                          SharedConfig config) {
        this.transactions = transactions;
        this.reviewLog = reviewLog;
        this.mlClient = mlClient;
        this.quickSignals = quickSignals;
        this.otpService = otpService;
        this.config = config;
    }

    /** What the caller gets back. */
    public record RoutingOutcome(Transaction transaction, OtpChallenge otpChallenge,
                                 List<String> reasons, String explanation) {
    }

    @Transactional
    public RoutingOutcome process(TransactionRequest request) {
        long started = System.currentTimeMillis();

        // ---- 1. STORE FIRST -------------------------------------------------
        // Persist before doing anything clever. If the ML call hangs or the JVM
        // dies mid-decision, we still have a record that money was moved -
        // losing the audit trail is worse than losing the score.
        Transaction txn = toEntity(request);
        txn = transactions.save(txn);

        // ---- 2. SCORE -------------------------------------------------------
        PredictionResult prediction = mlClient.predict(request);
        txn.setMlTransactionId(prediction.getMlTransactionId());
        txn.setFraudScore(prediction.getFraudScore());
        txn.setModelsDisagree(prediction.isModelsDisagree());
        txn.setDegradedMode(prediction.isDegraded());
        txn.setRuleApplied(prediction.getRuleApplied());

        List<String> reasons = new ArrayList<>(prediction.getReasons());

        Decision modelDecision = parse(prediction.getModelDecision());
        Decision current = parse(prediction.getDecision());
        txn.setModelDecision(modelDecision);

        reviewLog.save(new ReviewLog(txn.getId(),
                prediction.isDegraded() ? "SCORED_DEGRADED" : "SCORED",
                current.name(),
                String.format("score %.4f, models_disagree=%s%s",
                        prediction.getFraudScore(), prediction.isModelsDisagree(),
                        prediction.isDegraded()
                                ? ", DEGRADED: " + prediction.getDegradedReason() : "")));

        if (prediction.getRuleApplied() != null) {
            reviewLog.save(new ReviewLog(txn.getId(), "RULE_OVERRIDE", current.name(),
                    "Python rule '" + prediction.getRuleApplied()
                    + "' overrode the model decision of " + modelDecision));
        }

        // ---- 3. LOCAL RULES -------------------------------------------------
        // Java has the durable history in H2, so it can catch a burst that
        // spans a restart of the Python service.
        Decision afterRules = applyLocalRules(txn, current, reasons);
        if (afterRules.isMoreSevereThan(current)) {
            current = afterRules;
        }

        // ---- 4 & 5. ROUTE ---------------------------------------------------
        OtpChallenge challenge = null;
        String explanation;

        if (current == Decision.BLOCK) {
            txn.setFinalDecision(Decision.BLOCK);
            explanation = "Blocked automatically. " + firstReason(reasons);

        } else if (current == Decision.APPROVE && !prediction.isModelsDisagree()) {
            txn.setFinalDecision(Decision.APPROVE);
            // >>> Explain an approval with what CLEARED it. <<<
            // Using reasons.get(0) here printed "Approved automatically. the
            // sender's balance reconciles to the exact rupee, the signature of
            // an automated account drain" - the strongest WORRY, which the model
            // had weighed and dismissed, presented as the grounds for approval.
            // Verdict and evidence must point the same way.
            explanation = "Approved automatically. "
                    + firstReason(prediction.getProtectiveFactors().isEmpty()
                            ? List.of("nothing about it resembled known fraud")
                            : prediction.getProtectiveFactors());

        } else {
            // Uncertain: a middle score, or the two models disagreed.
            // Try to settle it cheaply before troubling the customer.
            QuickSignalService.Outcome signal = quickSignals.evaluate(txn);
            reasons.add(signal.detail());

            if (signal.resolved()) {
                txn.setQuickSignalApplied(signal.signal());
                Decision resolved = signal.decision();

                // A quick signal may only escalate - except that clearing an
                // uncertain payment via an established payee is precisely its
                // purpose, so APPROVE is allowed here and nowhere else.
                txn.setFinalDecision(resolved);
                explanation = "Settled without asking the customer: " + signal.detail();

                reviewLog.save(new ReviewLog(txn.getId(), "QUICK_SIGNAL",
                        resolved.name(), signal.signal() + " - " + signal.detail()));

            } else {
                txn.setFinalDecision(Decision.OTP_CONFIRM);
                txn = transactions.save(txn);
                challenge = otpService.sendChallenge(txn);
                explanation = "We have asked the customer to confirm this payment. "
                        + signal.detail();
            }
        }

        txn.setProcessingMillis(System.currentTimeMillis() - started);
        txn.setExplanation(explanation);
        txn.setReasonsJson(toJson(reasons));
        txn = transactions.save(txn);

        reviewLog.save(new ReviewLog(txn.getId(), "DECIDED",
                txn.getFinalDecision().name(),
                "Automatic decision, no human involved. " + explanation));

        log.info("Transaction {} -> {} (score {}, {}ms{})",
                txn.getId(), txn.getFinalDecision(), txn.getFraudScore(),
                txn.getProcessingMillis(),
                Boolean.TRUE.equals(txn.getDegradedMode()) ? ", DEGRADED" : "");

        return new RoutingOutcome(txn, challenge, reasons, explanation);
    }

    /**
     * Burst checks answered from H2.
     *
     * <p>These mirror rules.py, reading the same thresholds from config.yaml.
     * They exist on both sides deliberately: Python's copy uses its in-memory
     * history and is instant; Java's copy uses the durable database and
     * therefore survives a restart of the ML service. Either can catch a burst
     * the other misses.
     */
    private Decision applyLocalRules(Transaction txn, Decision current,
                                     List<String> reasons) {
        Decision result = current;

        Instant senderSince = Instant.now().minus(
                config.getRapidSenderWindowSteps(), ChronoUnit.HOURS);
        long sent = transactions.countSentSince(txn.getNameOrig(), senderSince, txn.getId());
        if (sent > config.getRapidSenderMaxAllowed()) {
            reasons.add("this account has made " + sent + " payments in the last "
                    + config.getRapidSenderWindowSteps() + " hour(s)");
            txn.setRuleApplied("rapid_sender_transfers");
            reviewLog.save(new ReviewLog(txn.getId(), "RULE_OVERRIDE", "BLOCK",
                    "rapid_sender_transfers: " + sent + " payments sent within "
                    + config.getRapidSenderWindowSteps() + "h, limit is "
                    + config.getRapidSenderMaxAllowed()));
            if (Decision.BLOCK.isMoreSevereThan(result)) {
                result = Decision.BLOCK;
            }
        }

        Instant receiverSince = Instant.now().minus(
                config.getRapidReceiverWindowSteps(), ChronoUnit.HOURS);
        long received = transactions.countReceivedSince(
                txn.getNameDest(), receiverSince, txn.getId());
        if (received > config.getRapidReceiverMaxAllowed()) {
            reasons.add("the receiving account has taken in " + received
                    + " payments in the last " + config.getRapidReceiverWindowSteps()
                    + " hour(s)");
            if (txn.getRuleApplied() == null) {
                txn.setRuleApplied("rapid_receiver_collection");
            }
            reviewLog.save(new ReviewLog(txn.getId(), "RULE_OVERRIDE", "OTP_CONFIRM",
                    "rapid_receiver_collection: " + received + " payments received "
                    + "within " + config.getRapidReceiverWindowSteps() + "h, limit is "
                    + config.getRapidReceiverMaxAllowed()));
            if (Decision.OTP_CONFIRM.isMoreSevereThan(result)) {
                result = Decision.OTP_CONFIRM;
            }
        }

        return result;
    }

    private Transaction toEntity(TransactionRequest r) {
        Transaction t = new Transaction();
        t.setType(r.getType() == null ? null : r.getType().toUpperCase());
        t.setAmount(r.getAmount());
        t.setNameOrig(r.getNameOrig());
        t.setOldbalanceOrg(r.getOldbalanceOrg());
        t.setNewbalanceOrig(r.getNewbalanceOrig());
        t.setNameDest(r.getNameDest());
        t.setOldbalanceDest(r.getOldbalanceDest());
        t.setNewbalanceDest(r.getNewbalanceDest());
        t.setStep(r.getStep());
        t.setLocation(r.getLocation());
        t.setLocationLat(r.getLocationLat());
        t.setLocationLon(r.getLocationLon());
        t.setTrustedDevice(Boolean.TRUE.equals(r.getTrustedDevice()));
        t.setReceivedAt(Instant.now());
        return t;
    }

    private static Decision parse(String value) {
        if (value == null) {
            return Decision.OTP_CONFIRM;
        }
        try {
            return Decision.valueOf(value.trim().toUpperCase());
        } catch (IllegalArgumentException e) {
            // An unrecognised decision from the ML service must not become an
            // approval by accident. Fall back to asking.
            log.warn("Unknown decision '{}' from ML service - defaulting to OTP", value);
            return Decision.OTP_CONFIRM;
        }
    }

    private static String firstReason(List<String> reasons) {
        return reasons.isEmpty() ? "" : reasons.get(0);
    }

    private String toJson(List<String> reasons) {
        try {
            return json.writeValueAsString(reasons);
        } catch (Exception e) {
            return "[]";
        }
    }
}

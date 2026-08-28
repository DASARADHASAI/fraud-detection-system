package com.klu.fraud.controller;

import com.klu.fraud.config.SharedConfig;
import com.klu.fraud.model.Decision;
import com.klu.fraud.model.OtpChallenge;
import com.klu.fraud.model.ReviewLog;
import com.klu.fraud.model.Account;
import com.klu.fraud.model.Transaction;
import com.klu.fraud.dto.TransactionRequest;
import com.klu.fraud.repository.ReviewLogRepository;
import com.klu.fraud.repository.TransactionRepository;
import com.klu.fraud.service.LedgerService;
import com.klu.fraud.service.MlServiceClient;
import com.klu.fraud.service.OtpService;
import com.klu.fraud.service.QuickSignalService;
import com.klu.fraud.service.RoutingService;
import jakarta.validation.Valid;
import java.time.Instant;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.springframework.data.domain.PageRequest;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

/**
 * The REST API the dashboard talks to.
 *
 * <p>Endpoints:
 * <pre>
 *   POST /api/transactions            submit a payment, get an automatic decision
 *   POST /api/transactions/{id}/otp   the customer answers the OTP challenge
 *   GET  /api/transactions            recent payments
 *   GET  /api/alerts                  everything that was not simply approved
 *   GET  /api/transactions/{id}       one payment plus its full audit trail
 *   GET  /api/transactions/{id}/explanation   the plain-language write-up
 *   POST /api/transactions/{id}/feedback      record what it really was
 *   GET  /api/stats                   dashboard summary counts
 *   GET  /api/health                  are both services alive
 * </pre>
 */
@RestController
@RequestMapping("/api")
@CrossOrigin
public class TransactionController {

    private final RoutingService routing;
    private final TransactionRepository transactions;
    private final ReviewLogRepository reviewLog;
    private final OtpService otpService;
    private final MlServiceClient mlClient;
    private final QuickSignalService quickSignals;
    private final SharedConfig config;
    private final LedgerService ledger;

    public TransactionController(RoutingService routing,
                                 TransactionRepository transactions,
                                 ReviewLogRepository reviewLog,
                                 OtpService otpService,
                                 MlServiceClient mlClient,
                                 QuickSignalService quickSignals,
                                 SharedConfig config,
                                 LedgerService ledger) {
        this.routing = routing;
        this.transactions = transactions;
        this.reviewLog = reviewLog;
        this.otpService = otpService;
        this.mlClient = mlClient;
        this.quickSignals = quickSignals;
        this.config = config;
        this.ledger = ledger;
    }

    /** Submit a payment. Comes back already decided - nothing waits for a human. */
    @PostMapping("/transactions")
    public ResponseEntity<Map<String, Object>> submit(
            @Valid @RequestBody TransactionRequest request) {

        applyLedgerBalances(request);

        RoutingService.RoutingOutcome outcome = routing.process(request);
        Transaction txn = outcome.transaction();

        if (txn.getFinalDecision() == Decision.APPROVE) {
            ledger.remember(txn.getNameDest(), "PAYMENT".equalsIgnoreCase(txn.getType()));
            ledger.settle(txn);
        }

        Map<String, Object> body = new LinkedHashMap<>();
        body.put("id", txn.getId());
        body.put("decision", txn.getFinalDecision());
        body.put("fraudScore", txn.getFraudScore());
        body.put("modelsDisagree", txn.getModelsDisagree());
        body.put("modelDecision", txn.getModelDecision());
        body.put("ruleApplied", txn.getRuleApplied());
        body.put("quickSignalApplied", txn.getQuickSignalApplied());
        body.put("degradedMode", txn.getDegradedMode());
        body.put("explanation", outcome.explanation());
        body.put("reasons", outcome.reasons());
        body.put("processingMillis", txn.getProcessingMillis());

        // The simulated OTP. In production the code would go by SMS and would
        // certainly NOT be returned in the API response.
        if (outcome.otpChallenge() != null) {
            OtpChallenge c = outcome.otpChallenge();
            Map<String, Object> otp = new LinkedHashMap<>();
            otp.put("challengeId", c.getId());
            otp.put("question", c.getQuestion());
            otp.put("code", c.getCode());
            otp.put("expiresAt", c.getExpiresAt());
            otp.put("timeoutSeconds", config.getOtpTimeoutSeconds());
            otp.put("note", "Simulated for the demo. No reply before the deadline "
                    + "is treated as a denial, so the payment would be blocked.");
            body.put("otp", otp);
        }

        return ResponseEntity.ok(body);
    }

    /**
     * Replace the balances the caller claimed with what the ledger actually says.
     *
     * <p>>>> WHY THE CALLER DOES NOT GET TO STATE ITS OWN BALANCE <<<
     * {@code amount_to_balance_ratio} and {@code orig_account_emptied} are the
     * 3rd and 4th strongest features in the model, and both are computed from
     * the sender's balance. If that number arrives in the request, an attacker
     * chooses the evidence used to judge them: claim a huge opening balance and
     * a drain stops looking like a drain.
     *
     * <p>So for any account this bank actually holds, the request's balances are
     * overwritten from the ledger. Accounts the bank does not hold - which is
     * every counterparty at another bank, and every account the test suites
     * invent - are left exactly as sent, so this is additive and changes no
     * existing behaviour.
     */
    private void applyLedgerBalances(TransactionRequest request) {
        Optional<Account> payer = ledger.find(request.getNameOrig());
        if (payer.isPresent()) {
            double before = payer.get().getBalance();
            request.setOldbalanceOrg(before);
            request.setNewbalanceOrig(Math.max(0.0, before - request.getAmount()));
        }

        Optional<Account> payee = ledger.find(request.getNameDest());
        if (payee.isPresent()) {
            double before = payee.get().getBalance();
            request.setOldbalanceDest(before);
            request.setNewbalanceDest(before + request.getAmount());
        }
    }

    /** The customer answers: YES it was me, or NO it was not. */
    @PostMapping("/transactions/{id}/otp")
    public ResponseEntity<Map<String, Object>> answerOtp(
            @PathVariable Long id,
            @RequestParam(name = "confirmed") boolean confirmed) {
        try {
            Decision decision = otpService.respond(id, confirmed);

            // The customer said YES. That is the one piece of evidence nothing
            // else in the system has, and it is the only thing allowed to move
            // a decision toward APPROVE - so the money moves here, not earlier.
            if (decision == Decision.APPROVE) {
                transactions.findById(id).ifPresent(t -> {
                    ledger.remember(t.getNameDest(), "PAYMENT".equalsIgnoreCase(t.getType()));
                    ledger.settle(t);
                });
            }

            Map<String, Object> body = new LinkedHashMap<>();
            body.put("id", id);
            body.put("customerAnswered", confirmed ? "YES" : "NO");
            body.put("decision", decision);
            body.put("explanation", confirmed
                    ? "The customer confirmed the payment, so it was approved."
                    : "The customer denied the payment, so it was blocked.");
            return ResponseEntity.ok(body);
        } catch (IllegalStateException e) {
            return ResponseEntity.status(HttpStatus.CONFLICT)
                    .body(Map.of("error", e.getMessage()));
        }
    }

    @GetMapping("/transactions")
    public List<Transaction> recent(@RequestParam(defaultValue = "50") int limit) {
        return transactions.findAllByOrderByReceivedAtDesc(
                PageRequest.of(0, Math.min(limit, 500)));
    }

    /** Everything that was not simply approved - what the dashboard highlights. */
    @GetMapping("/alerts")
    public List<Transaction> alerts(@RequestParam(defaultValue = "50") int limit) {
        return transactions.findAlerts(PageRequest.of(0, Math.min(limit, 500)));
    }

    /** One payment plus its full audit trail. */
    @GetMapping("/transactions/{id}")
    public ResponseEntity<Map<String, Object>> detail(@PathVariable Long id) {
        return transactions.findById(id)
                .<ResponseEntity<Map<String, Object>>>map(txn -> {
                    List<ReviewLog> trail =
                            reviewLog.findByTransactionIdOrderByLoggedAtAsc(id);
                    Map<String, Object> body = new LinkedHashMap<>();
                    body.put("transaction", txn);
                    body.put("reviewLog", trail);
                    body.put("otp", otpService.findPending(id).orElse(null));
                    return ResponseEntity.ok(body);
                })
                .orElseGet(() -> ResponseEntity.status(HttpStatus.NOT_FOUND)
                        .body(Map.of("error", "no transaction with id " + id)));
    }

    /**
     * The plain-language write-up, fetched lazily from the Python service.
     *
     * <p>Deliberately NOT part of the submit response: the language model takes
     * about 3.7 seconds, against 28 milliseconds for scoring. A payment cannot
     * wait for prose nobody may read.
     */
    @GetMapping("/transactions/{id}/explanation")
    public ResponseEntity<Object> explanation(@PathVariable Long id) {
        return transactions.findById(id)
                .<ResponseEntity<Object>>map(txn -> {
                    if (txn.getMlTransactionId() == null) {
                        return ResponseEntity.ok(Map.of(
                                "explanation", "No detailed explanation is available: "
                                        + "this payment was decided while the scoring "
                                        + "service was unavailable.",
                                "source", "degraded"));
                    }
                    return ResponseEntity.ok(
                            mlClient.fetchExplanation(txn.getMlTransactionId()));
                })
                .orElseGet(() -> ResponseEntity.status(HttpStatus.NOT_FOUND)
                        .body(Map.of("error", "no transaction with id " + id)));
    }

    /**
     * Record what the payment really turned out to be.
     *
     * <p>Stored for future model improvement. Nothing retrains on it today -
     * that is future scope, and saying so is more honest than implying a
     * feedback loop that does not exist.
     */
    @PostMapping("/transactions/{id}/feedback")
    public ResponseEntity<Map<String, Object>> feedback(
            @PathVariable Long id,
            @RequestParam boolean wasFraud,
            @RequestParam(required = false) String note) {

        return transactions.findById(id)
                .<ResponseEntity<Map<String, Object>>>map(txn -> {
                    txn.setConfirmedFraud(wasFraud);
                    txn.setFeedbackNote(note);
                    txn.setFeedbackAt(Instant.now());
                    transactions.save(txn);
                    reviewLog.save(new ReviewLog(id, "FEEDBACK",
                            txn.getFinalDecision().name(),
                            "Outcome recorded: " + (wasFraud ? "FRAUD" : "legitimate")
                            + (note == null ? "" : " - " + note)));
                    return ResponseEntity.ok(Map.of(
                            "id", id,
                            "confirmedFraud", wasFraud,
                            "note", "Saved for future model improvement (future scope "
                                    + "- no retraining happens automatically)."));
                })
                .orElseGet(() -> ResponseEntity.status(HttpStatus.NOT_FOUND)
                        .body(Map.of("error", "no transaction with id " + id)));
    }

    @GetMapping("/stats")
    public Map<String, Object> stats() {
        long approved = transactions.countByFinalDecision(Decision.APPROVE);
        long otp = transactions.countByFinalDecision(Decision.OTP_CONFIRM);
        long blocked = transactions.countByFinalDecision(Decision.BLOCK);
        long total = approved + otp + blocked;

        Map<String, Object> body = new LinkedHashMap<>();
        body.put("total", total);
        body.put("approved", approved);
        body.put("otpConfirm", otp);
        body.put("blocked", blocked);
        body.put("approveBelow", config.getApproveBelow());
        body.put("blockAbove", config.getBlockAbove());
        body.put("humanReviewers", 0);   // and there never will be
        return body;
    }

    @GetMapping("/health")
    public Map<String, Object> health() {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("backend", "ok");
        body.put("mlService", mlClient.isMlServiceHealthy() ? "ok" : "unreachable");
        body.put("mlServiceUrl", config.getMlServiceUrl());
        body.put("configLoadedFrom", config.getConfigPath());
        body.put("thresholds", Map.of(
                "approveBelow", config.getApproveBelow(),
                "blockAbove", config.getBlockAbove(),
                "otpTimeoutSeconds", config.getOtpTimeoutSeconds()));
        body.put("knownFraudAccounts", quickSignals.getKnownFraudAccounts());
        return body;
    }

    /** Turn validation failures into a readable list instead of a stack trace. */
    @ExceptionHandler(MethodArgumentNotValidException.class)
    public ResponseEntity<Map<String, Object>> onValidationError(
            MethodArgumentNotValidException e) {
        Map<String, Object> errors = new HashMap<>();
        e.getBindingResult().getFieldErrors().forEach(fieldError ->
                errors.put(fieldError.getField(), fieldError.getDefaultMessage()));
        return ResponseEntity.badRequest().body(Map.of(
                "error", "invalid transaction", "details", errors));
    }
}

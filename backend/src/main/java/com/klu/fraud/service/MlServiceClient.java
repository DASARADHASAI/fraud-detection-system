package com.klu.fraud.service;

import com.klu.fraud.config.SharedConfig;
import com.klu.fraud.dto.PredictionResult;
import com.klu.fraud.dto.TransactionRequest;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.web.client.RestTemplateBuilder;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

/**
 * Talks to the Python ML service.
 *
 * <p>>>> THE INTERESTING PART OF THIS CLASS IS WHAT HAPPENS WHEN PYTHON IS DOWN. <<<
 *
 * <p>A fraud system that stops working when one dependency dies is not a fraud
 * system, it is an outage. But the two obvious fallbacks are both wrong:
 *
 * <ul>
 *   <li><b>Approve everything</b> - the bank is now completely unprotected, and
 *       an attacker who can knock over one service gets a free pass. This is
 *       the failure mode that costs real money.</li>
 *   <li><b>Block everything</b> - every customer's card stops working. The bank
 *       has effectively DDoSed itself, and the support lines melt.</li>
 * </ul>
 *
 * <p>So we do neither. When Python is unreachable we apply a deliberately crude
 * local heuristic and send anything remotely suspicious to OTP confirmation.
 * The customer becomes the fraud model. It is worse than the real thing - the
 * heuristic knows nothing about balance-error patterns or receiver history -
 * but it fails towards ASKING rather than towards guessing, and every decision
 * it makes is flagged {@code degraded} all the way to the dashboard.
 */
@Service
public class MlServiceClient {

    private static final Logger log = LoggerFactory.getLogger(MlServiceClient.class);

    private final SharedConfig config;
    private final RestTemplate restTemplate;

    public MlServiceClient(SharedConfig config, RestTemplateBuilder builder) {
        this.config = config;
        // A timeout is not optional here. Without one, a hung Python process
        // holds a Tomcat thread open until the socket gives up, and a handful
        // of payments can exhaust the whole pool.
        this.restTemplate = builder
                .connectTimeout(Duration.ofSeconds(config.getMlTimeoutSeconds()))
                .readTimeout(Duration.ofSeconds(config.getMlTimeoutSeconds()))
                .build();
    }

    /** Score a payment. Never throws - returns a degraded result instead. */
    @SuppressWarnings("unchecked")
    public PredictionResult predict(TransactionRequest request) {
        String url = config.getMlServiceUrl() + "/predict";

        Map<String, Object> body = new HashMap<>();
        body.put("type", request.getType());
        body.put("amount", request.getAmount());
        body.put("nameOrig", request.getNameOrig());
        body.put("oldbalanceOrg", request.getOldbalanceOrg());
        body.put("newbalanceOrig", request.getNewbalanceOrig());
        body.put("nameDest", request.getNameDest());
        body.put("oldbalanceDest", request.getOldbalanceDest());
        body.put("newbalanceDest", request.getNewbalanceDest());
        if (request.getStep() != null) body.put("step", request.getStep());
        if (request.getLocation() != null) body.put("location", request.getLocation());
        if (request.getLocationLat() != null) body.put("location_lat", request.getLocationLat());
        if (request.getLocationLon() != null) body.put("location_lon", request.getLocationLon());
        body.put("trusted_device", Boolean.TRUE.equals(request.getTrustedDevice()));

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);

        try {
            Map<String, Object> response =
                    restTemplate.postForObject(url, new HttpEntity<>(body, headers), Map.class);
            if (response == null) {
                return degraded(request, "ML service returned an empty body");
            }

            PredictionResult result = new PredictionResult();
            result.setMlTransactionId(str(response.get("transaction_id")));
            result.setFraudScore(dbl(response.get("fraud_score"), 0.5));
            result.setModelsDisagree(Boolean.TRUE.equals(response.get("models_disagree")));
            result.setDecision(str(response.getOrDefault("decision", "OTP_CONFIRM")));
            result.setModelDecision(str(response.getOrDefault("model_decision",
                    response.getOrDefault("decision", "OTP_CONFIRM"))));
            result.setRuleApplied(str(response.get("rule_applied")));
            // rules.py writes a customer_message per triggered rule; use it so a
            // rule-driven decision is explained by the rule, not by a model feature.
            Object triggered = response.get("rules_triggered");
            if (triggered instanceof List<?> list && !list.isEmpty()
                    && list.get(0) instanceof Map<?, ?> first) {
                Object message = first.get("customer_message");
                if (message != null) {
                    result.setRuleMessage(String.valueOf(message));
                }
            }

            List<String> reasons = extractReasons(response.get("shap_reasons"));
            result.setReasons(reasons);
            result.setProtectiveFactors(extractReasons(response.get("protective_factors")));
            return result;

        } catch (Exception e) {
            log.error("ML service unreachable at {} ({}: {}). Falling back.",
                    url, e.getClass().getSimpleName(), e.getMessage());
            return degraded(request, e.getClass().getSimpleName() + ": " + e.getMessage());
        }
    }

    /**
     * The local safety net.
     *
     * <p>Crude on purpose - it uses only what is visible in the payment itself,
     * with no model, no history and no training. It exists to keep the bank
     * running for the minutes Python is restarting, not to replace it.
     *
     * <p>Note the bias: this errs towards OTP_CONFIRM, never towards APPROVE.
     * When the system cannot see clearly, it asks the customer rather than
     * guessing on their behalf.
     */
    private PredictionResult degraded(TransactionRequest request, String why) {
        PredictionResult result = new PredictionResult();
        result.setDegraded(true);
        result.setDegradedReason(why);
        result.setMlTransactionId(null);

        List<String> reasons = new ArrayList<>();
        reasons.add("The fraud scoring service was unavailable, so this payment "
                + "was judged by a basic backup check only");

        double amount = request.getAmount() == null ? 0 : request.getAmount();
        double oldBalance = request.getOldbalanceOrg() == null ? 0 : request.getOldbalanceOrg();
        double newBalance = request.getNewbalanceOrig() == null ? 0 : request.getNewbalanceOrig();
        String type = request.getType() == null ? "" : request.getType().toUpperCase();

        boolean riskyType = type.equals("TRANSFER") || type.equals("CASH_OUT");
        boolean drained = newBalance == 0 && oldBalance > 0 && amount > 0;
        boolean movesMostOfBalance = oldBalance > 0 && (amount / oldBalance) > 0.90;
        boolean receiverWasEmpty = request.getOldbalanceDest() != null
                && request.getOldbalanceDest() == 0;

        int concerns = 0;
        if (riskyType) {
            concerns++;
            reasons.add("this is a TRANSFER or CASH_OUT");
        }
        if (drained) {
            concerns++;
            reasons.add("this payment empties the sending account completely");
        }
        if (movesMostOfBalance) {
            concerns++;
            reasons.add("it moves almost the whole balance");
        }
        if (receiverWasEmpty) {
            concerns++;
            reasons.add("the receiving account was empty beforehand");
        }

        // A rough stand-in score, so the dashboard still has something to show.
        // It is NOT a model output and must never be presented as one.
        result.setFraudScore(Math.min(0.25 * concerns, 0.99));

        if (drained && riskyType && concerns >= 3) {
            // The clearest account-takeover shape there is. Even a crude check
            // should stop this rather than bother the customer with a question.
            result.setDecision("BLOCK");
            result.setModelDecision("BLOCK");
        } else if (concerns >= 1) {
            result.setDecision("OTP_CONFIRM");
            result.setModelDecision("OTP_CONFIRM");
        } else {
            result.setDecision("APPROVE");
            result.setModelDecision("APPROVE");
            reasons.add("nothing obvious was wrong with it");
        }

        result.setReasons(reasons);
        return result;
    }

    /**
     * Fetch the plain-language write-up for a payment already scored.
     *
     * <p>Slow - the language model takes seconds - which is exactly why this is
     * a separate call made only when someone opens a transaction, rather than
     * part of the scoring path.
     *
     * <p>Uses a longer timeout than {@link #predict}: the normal ML timeout is
     * sized for a 28 ms scoring call and would abort a 3.7 s narration every
     * single time.
     */
    @SuppressWarnings("unchecked")
    public Map<String, Object> fetchExplanation(String mlTransactionId) {
        String url = config.getMlServiceUrl() + "/explain";
        try {
            RestTemplate slowTemplate = new RestTemplateBuilder()
                    .connectTimeout(Duration.ofSeconds(5))
                    .readTimeout(Duration.ofSeconds(60))
                    .build();

            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            Map<String, Object> body = Map.of("transaction_id", mlTransactionId);

            Map<String, Object> response = slowTemplate.postForObject(
                    url, new HttpEntity<>(body, headers), Map.class);
            return response == null ? Map.of("error", "empty response") : response;

        } catch (Exception e) {
            log.warn("Could not fetch explanation for {}: {}", mlTransactionId,
                    e.getMessage());
            return Map.of(
                    "explanation", "The detailed explanation could not be loaded "
                            + "because the scoring service is unavailable. The "
                            + "decision itself is unaffected - it was already made.",
                    "source", "unavailable",
                    "error", e.getClass().getSimpleName());
        }
    }

    /** Is Python up? Used by /api/health so the dashboard can show a banner. */
    public boolean isMlServiceHealthy() {
        try {
            restTemplate.getForObject(config.getMlServiceUrl() + "/health", Map.class);
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    /** Pull the "reason" strings out of a list of SHAP factor objects. */
    private static List<String> extractReasons(Object raw) {
        List<String> out = new ArrayList<>();
        if (raw instanceof List<?> list) {
            for (Object item : list) {
                if (item instanceof Map<?, ?> map && map.get("reason") != null) {
                    out.add(String.valueOf(map.get("reason")));
                }
            }
        }
        return out;
    }

    private static String str(Object value) {
        return value == null ? null : String.valueOf(value);
    }

    private static double dbl(Object value, double fallback) {
        return value instanceof Number n ? n.doubleValue() : fallback;
    }
}

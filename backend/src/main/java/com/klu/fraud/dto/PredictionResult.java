package com.klu.fraud.dto;

import java.util.ArrayList;
import java.util.List;

/**
 * What the Python service told us about a payment - or, when Python is
 * unreachable, what our local safety net concluded instead.
 *
 * <p>The {@code degraded} flag is the important field. It travels all the way
 * to the dashboard so nobody can mistake a fallback decision for a
 * model-backed one.
 */
public class PredictionResult {

    private String mlTransactionId;
    private double fraudScore;
    private boolean modelsDisagree;
    private String decision = "OTP_CONFIRM";
    private String modelDecision = "OTP_CONFIRM";
    private String ruleApplied;
    /**
     * The customer-facing sentence for the rule that fired, written in
     * rules.py. Needed because a rule-driven block must be explained by what
     * the RULE found - quoting a model feature instead would describe a
     * decision the model did not make.
     */
    private String ruleMessage;
    private List<String> reasons = new ArrayList<>();
    /**
     * Why the model was RELAXED about this payment. Needed because an approval
     * must be explained by what cleared it, not by the strongest worry that was
     * outweighed - see RoutingService.explain().
     */
    private List<String> protectiveFactors = new ArrayList<>();
    private boolean degraded = false;
    private String degradedReason;

    public String getMlTransactionId() { return mlTransactionId; }
    public void setMlTransactionId(String v) { this.mlTransactionId = v; }

    public double getFraudScore() { return fraudScore; }
    public void setFraudScore(double v) { this.fraudScore = v; }

    public boolean isModelsDisagree() { return modelsDisagree; }
    public void setModelsDisagree(boolean v) { this.modelsDisagree = v; }

    public String getDecision() { return decision; }
    public void setDecision(String v) { this.decision = v; }

    public String getModelDecision() { return modelDecision; }
    public void setModelDecision(String v) { this.modelDecision = v; }

    public String getRuleApplied() { return ruleApplied; }
    public void setRuleApplied(String v) { this.ruleApplied = v; }

    public String getRuleMessage() { return ruleMessage; }
    public void setRuleMessage(String v) { this.ruleMessage = v; }

    public List<String> getReasons() { return reasons; }
    public void setReasons(List<String> v) { this.reasons = v; }

    public List<String> getProtectiveFactors() { return protectiveFactors; }
    public void setProtectiveFactors(List<String> v) { this.protectiveFactors = v; }

    public boolean isDegraded() { return degraded; }
    public void setDegraded(boolean v) { this.degraded = v; }

    public String getDegradedReason() { return degradedReason; }
    public void setDegradedReason(String v) { this.degradedReason = v; }
}

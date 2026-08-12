package com.klu.fraud.model;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Index;
import jakarta.persistence.Lob;
import jakarta.persistence.Table;
import java.time.Instant;

/**
 * One payment, plus everything the system decided about it.
 *
 * <p>Stored in H2 in FILE mode, so the table survives a restart and the demo
 * keeps its history. Indexed on the accounts because the quick-signal checks
 * and the rule layer both query by account on every single payment.
 */
@Entity
@Table(name = "transactions", indexes = {
        @Index(name = "idx_name_orig", columnList = "nameOrig"),
        @Index(name = "idx_name_dest", columnList = "nameDest"),
        @Index(name = "idx_received_at", columnList = "receivedAt")
})
public class Transaction {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    /** The id the Python service gave this scoring result, for /explain later. */
    private String mlTransactionId;

    // ---- The payment itself -------------------------------------------------
    private Integer step;
    private String type;
    private Double amount;
    private String nameOrig;
    private Double oldbalanceOrg;
    private Double newbalanceOrig;
    private String nameDest;
    private Double oldbalanceDest;
    private Double newbalanceDest;

    /** Simulated, for the impossible-travel demo. See config.yaml. */
    private String location;
    private Double locationLat;
    private Double locationLon;

    /**
     * Whether the payment came from a device this customer has used before.
     * NOTE this is never sufficient on its own to clear a payment - see
     * QuickSignalService and knowledge_base case FRAUD-006.
     */
    private Boolean trustedDevice = false;

    // ---- What the model said ------------------------------------------------
    private Double fraudScore;
    private Boolean modelsDisagree = false;

    @Enumerated(EnumType.STRING)
    private Decision modelDecision;

    // ---- What actually happened --------------------------------------------
    @Enumerated(EnumType.STRING)
    private Decision finalDecision;

    /** Which rule overrode the model, if any. Null when the model stood. */
    private String ruleApplied;

    /** Which quick signal resolved a borderline payment, if any. */
    private String quickSignalApplied;

    /**
     * True when the Python service could not be reached and we fell back to
     * the local safety net. Surfaced on the dashboard so nobody mistakes a
     * degraded decision for a model-backed one.
     */
    private Boolean degradedMode = false;

    /**
     * The one-sentence explanation shown to the customer, already matched to the
     * decision that was taken.
     *
     * <p>Persisted rather than rebuilt in the browser. The dashboard's feed used
     * to render reasonsJson[0], which is the strongest RISK factor - so an
     * approved payment was captioned "this is a TRANSFER or CASH_OUT, the only
     * two types where fraud occurs". Storing the sentence keeps the feed and the
     * verdict panel saying the same thing.
     */
    @Column(length = 1000)
    private String explanation;

    /** Human-readable reasons, stored as JSON so the dashboard can list them. */
    @Lob
    @Column(length = 4000)
    private String reasonsJson;

    private Instant receivedAt = Instant.now();
    private Long processingMillis;

    // ---- Feedback for future model improvement (future scope) ---------------
    /** What it actually turned out to be, if anyone ever tells us. */
    private Boolean confirmedFraud;
    private String feedbackNote;
    private Instant feedbackAt;

    public Transaction() {
    }

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }

    public String getMlTransactionId() { return mlTransactionId; }
    public void setMlTransactionId(String v) { this.mlTransactionId = v; }

    public Integer getStep() { return step; }
    public void setStep(Integer step) { this.step = step; }

    public String getType() { return type; }
    public void setType(String type) { this.type = type; }

    public Double getAmount() { return amount; }
    public void setAmount(Double amount) { this.amount = amount; }

    public String getNameOrig() { return nameOrig; }
    public void setNameOrig(String v) { this.nameOrig = v; }

    public Double getOldbalanceOrg() { return oldbalanceOrg; }
    public void setOldbalanceOrg(Double v) { this.oldbalanceOrg = v; }

    public Double getNewbalanceOrig() { return newbalanceOrig; }
    public void setNewbalanceOrig(Double v) { this.newbalanceOrig = v; }

    public String getNameDest() { return nameDest; }
    public void setNameDest(String v) { this.nameDest = v; }

    public Double getOldbalanceDest() { return oldbalanceDest; }
    public void setOldbalanceDest(Double v) { this.oldbalanceDest = v; }

    public Double getNewbalanceDest() { return newbalanceDest; }
    public void setNewbalanceDest(Double v) { this.newbalanceDest = v; }

    public String getLocation() { return location; }
    public void setLocation(String v) { this.location = v; }

    public Double getLocationLat() { return locationLat; }
    public void setLocationLat(Double v) { this.locationLat = v; }

    public Double getLocationLon() { return locationLon; }
    public void setLocationLon(Double v) { this.locationLon = v; }

    public Boolean getTrustedDevice() { return trustedDevice; }
    public void setTrustedDevice(Boolean v) { this.trustedDevice = v; }

    public Double getFraudScore() { return fraudScore; }
    public void setFraudScore(Double v) { this.fraudScore = v; }

    public Boolean getModelsDisagree() { return modelsDisagree; }
    public void setModelsDisagree(Boolean v) { this.modelsDisagree = v; }

    public Decision getModelDecision() { return modelDecision; }
    public void setModelDecision(Decision v) { this.modelDecision = v; }

    public Decision getFinalDecision() { return finalDecision; }
    public void setFinalDecision(Decision v) { this.finalDecision = v; }

    public String getRuleApplied() { return ruleApplied; }
    public void setRuleApplied(String v) { this.ruleApplied = v; }

    public String getQuickSignalApplied() { return quickSignalApplied; }
    public void setQuickSignalApplied(String v) { this.quickSignalApplied = v; }

    public Boolean getDegradedMode() { return degradedMode; }
    public void setDegradedMode(Boolean v) { this.degradedMode = v; }

    public String getExplanation() { return explanation; }
    public void setExplanation(String v) { this.explanation = v; }

    public String getReasonsJson() { return reasonsJson; }
    public void setReasonsJson(String v) { this.reasonsJson = v; }

    public Instant getReceivedAt() { return receivedAt; }
    public void setReceivedAt(Instant v) { this.receivedAt = v; }

    public Long getProcessingMillis() { return processingMillis; }
    public void setProcessingMillis(Long v) { this.processingMillis = v; }

    public Boolean getConfirmedFraud() { return confirmedFraud; }
    public void setConfirmedFraud(Boolean v) { this.confirmedFraud = v; }

    public String getFeedbackNote() { return feedbackNote; }
    public void setFeedbackNote(String v) { this.feedbackNote = v; }

    public Instant getFeedbackAt() { return feedbackAt; }
    public void setFeedbackAt(Instant v) { this.feedbackAt = v; }
}

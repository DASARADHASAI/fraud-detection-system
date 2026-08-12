package com.klu.fraud.model;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Index;
import jakarta.persistence.Lob;
import jakarta.persistence.Table;
import java.time.Instant;

/**
 * The audit trail.
 *
 * <p>>>> READ THIS BEFORE ASSUMING IT IS A REVIEW QUEUE. <<<
 * Nothing waits here. Every payment is already decided by the time a row is
 * written. This is a RECORD of automatic decisions that the bank COULD inspect
 * afterwards - for a regulator, a complaint, or a post-mortem - not a list of
 * work for a human to do. The system has no human step anywhere.
 *
 * <p>One row per event in a payment's life, so the full story is reconstructable:
 * what the model said, what a rule did to it, what the customer answered.
 */
@Entity
@Table(name = "review_log", indexes = {
        @Index(name = "idx_log_txn", columnList = "transactionId"),
        @Index(name = "idx_log_at", columnList = "loggedAt")
})
public class ReviewLog {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private Long transactionId;

    /** SCORED, RULE_OVERRIDE, QUICK_SIGNAL, OTP_SENT, OTP_ANSWERED, FEEDBACK... */
    private String event;

    private String decisionAtThisPoint;

    @Lob
    @Column(length = 2000)
    private String detail;

    private Instant loggedAt = Instant.now();

    public ReviewLog() {
    }

    public ReviewLog(Long transactionId, String event, String decisionAtThisPoint,
                     String detail) {
        this.transactionId = transactionId;
        this.event = event;
        this.decisionAtThisPoint = decisionAtThisPoint;
        this.detail = detail;
    }

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }

    public Long getTransactionId() { return transactionId; }
    public void setTransactionId(Long v) { this.transactionId = v; }

    public String getEvent() { return event; }
    public void setEvent(String v) { this.event = v; }

    public String getDecisionAtThisPoint() { return decisionAtThisPoint; }
    public void setDecisionAtThisPoint(String v) { this.decisionAtThisPoint = v; }

    public String getDetail() { return detail; }
    public void setDetail(String v) { this.detail = v; }

    public Instant getLoggedAt() { return loggedAt; }
    public void setLoggedAt(Instant v) { this.loggedAt = v; }
}

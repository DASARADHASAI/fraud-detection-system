package com.klu.fraud.model;

import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Index;
import jakarta.persistence.Table;
import java.time.Instant;

/**
 * A "did you make this payment?" challenge sent to the customer.
 *
 * <p>This is the entire step-up mechanism, and it is what replaces the human
 * reviewer. Where a traditional system would queue an uncertain payment for an
 * analyst, this system asks the one person who actually knows: the customer.
 *
 * <p>The OTP is SIMULATED for the demo - no SMS is sent. The code is returned
 * in the API response so the dashboard can display it.
 */
@Entity
@Table(name = "otp_challenges", indexes = {
        @Index(name = "idx_otp_txn", columnList = "transactionId"),
        @Index(name = "idx_otp_status", columnList = "status")
})
public class OtpChallenge {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private Long transactionId;
    private String code;
    private String customerAccount;

    @Enumerated(EnumType.STRING)
    private OtpStatus status = OtpStatus.PENDING;

    private Instant sentAt = Instant.now();
    private Instant expiresAt;
    private Instant respondedAt;

    /** The question the customer sees. Kept short and unambiguous on purpose. */
    private String question;

    public OtpChallenge() {
    }

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }

    public Long getTransactionId() { return transactionId; }
    public void setTransactionId(Long v) { this.transactionId = v; }

    public String getCode() { return code; }
    public void setCode(String v) { this.code = v; }

    public String getCustomerAccount() { return customerAccount; }
    public void setCustomerAccount(String v) { this.customerAccount = v; }

    public OtpStatus getStatus() { return status; }
    public void setStatus(OtpStatus v) { this.status = v; }

    public Instant getSentAt() { return sentAt; }
    public void setSentAt(Instant v) { this.sentAt = v; }

    public Instant getExpiresAt() { return expiresAt; }
    public void setExpiresAt(Instant v) { this.expiresAt = v; }

    public Instant getRespondedAt() { return respondedAt; }
    public void setRespondedAt(Instant v) { this.respondedAt = v; }

    public String getQuestion() { return question; }
    public void setQuestion(String v) { this.question = v; }

    public boolean isExpired() {
        return status == OtpStatus.PENDING
                && expiresAt != null
                && Instant.now().isAfter(expiresAt);
    }
}

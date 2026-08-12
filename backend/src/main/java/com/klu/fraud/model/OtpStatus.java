package com.klu.fraud.model;

/** Where an OTP challenge has got to. */
public enum OtpStatus {
    /** Sent to the customer, still waiting for a reply. */
    PENDING,
    /** Customer said "yes, that was me" -> the payment is approved. */
    CONFIRMED,
    /** Customer said "no, that was not me" -> the payment is blocked. */
    DENIED,
    /**
     * No reply before the deadline. Treated exactly like a denial.
     * Silence is not consent when someone's money is moving.
     */
    TIMED_OUT
}

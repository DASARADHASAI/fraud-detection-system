package com.klu.fraud.model;

/**
 * Every possible outcome for a payment. There are only three, and a human
 * reviewer is not one of them - every value here is reached automatically.
 */
public enum Decision {
    /** Score below the approve threshold and nothing else objected. */
    APPROVE,

    /**
     * Uncertain. The customer is asked to confirm by OTP. Their answer decides:
     * confirm -> APPROVE, deny or no reply before the timeout -> BLOCK.
     */
    OTP_CONFIRM,

    /** Stopped automatically. */
    BLOCK;

    /**
     * How severe each outcome is. Used to guarantee that rules and quick
     * signals can only ever escalate a decision, never soften one.
     */
    public int severity() {
        return switch (this) {
            case APPROVE -> 0;
            case OTP_CONFIRM -> 1;
            case BLOCK -> 2;
        };
    }

    public boolean isMoreSevereThan(Decision other) {
        return this.severity() > other.severity();
    }
}

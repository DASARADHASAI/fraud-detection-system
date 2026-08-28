package com.klu.fraud.model;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Index;
import jakarta.persistence.Table;
import java.time.Instant;

/**
 * A device this customer has actually used before.
 *
 * <p>>>> WHY THIS IS NOT A CHECKBOX <<<
 * "Known device" was previously a switch on the UniPay page, which meant it
 * asserted trust rather than evidencing it. Here trust is <b>earned</b>: the
 * first time a browser is seen it is NOT trusted, and it becomes known only
 * because the bank has a record of having seen it before.
 *
 * <p>That is the real property. A device is trusted because of history, and
 * history is something an attacker on a fresh machine does not have.
 *
 * <p>Note what this still does not prove. A known device establishes
 * <i>identity</i>, never <i>intent</i> - see knowledge-base case FRAUD-006,
 * where the victim authorises the payment themselves from their own phone.
 * That is why {@code QuickSignalService} requires an established payee
 * alongside it and never clears a payment on the device alone.
 */
@Entity
@Table(name = "known_devices", indexes = {
        @Index(name = "idx_device_account", columnList = "accountHandle")
})
public class KnownDevice {

    /** Browser fingerprint: a localStorage id bound to coarse device traits. */
    @Id
    @Column(length = 128)
    private String fingerprint;

    @Column(length = 128)
    private String accountHandle;

    private Instant firstSeen = Instant.now();
    private Instant lastSeen = Instant.now();

    /** How many times this device has been seen. 1 means "right now, first time". */
    private Integer seenCount = 1;

    protected KnownDevice() { }

    public KnownDevice(String fingerprint, String accountHandle) {
        this.fingerprint = fingerprint;
        this.accountHandle = accountHandle;
        this.firstSeen = Instant.now();
        this.lastSeen = this.firstSeen;
        this.seenCount = 1;
    }

    public String getFingerprint() { return fingerprint; }
    public void setFingerprint(String v) { this.fingerprint = v; }

    public String getAccountHandle() { return accountHandle; }
    public void setAccountHandle(String v) { this.accountHandle = v; }

    public Instant getFirstSeen() { return firstSeen; }
    public void setFirstSeen(Instant v) { this.firstSeen = v; }

    public Instant getLastSeen() { return lastSeen; }
    public void setLastSeen(Instant v) { this.lastSeen = v; }

    public Integer getSeenCount() { return seenCount; }
    public void setSeenCount(Integer v) { this.seenCount = v; }
}

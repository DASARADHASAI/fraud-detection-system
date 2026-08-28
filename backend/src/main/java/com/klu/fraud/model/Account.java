package com.klu.fraud.model;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import java.time.Instant;

/**
 * A real account, with a real balance, held by the bank.
 *
 * <p>>>> WHY THIS TABLE EXISTS <<<
 * UniPay used to hold its balance and its payee list in JavaScript literals.
 * That is fine for a picture and wrong for a demonstration: anyone opening
 * View Source sees the numbers were asserted by the page rather than looked up.
 *
 * <p>More importantly it was a security lie. A client-asserted balance is
 * forgeable - a payer could claim any opening balance they liked, and
 * {@code amount_to_balance_ratio} and {@code orig_account_emptied} are the 3rd
 * and 4th strongest features in the model. A fraud engine that reads its most
 * important evidence from the attacker's own request is not a fraud engine.
 *
 * <p>So the balance lives here, on the bank's side, and
 * {@code TransactionController} overrides whatever the caller claimed with what
 * the ledger actually says.
 */
@Entity
@Table(name = "accounts")
public class Account {

    /** The UPI address. This is the account identifier everywhere else too. */
    @Id
    @Column(length = 128)
    private String handle;

    private String displayName;

    /** Rupees. Moved only by {@code LedgerService.settle}, never by a client. */
    private Double balance = 0.0;

    /** A shop rather than a person - maps to PAYMENT instead of TRANSFER. */
    private Boolean merchant = false;

    private Instant openedAt = Instant.now();

    protected Account() { }

    public Account(String handle, String displayName, double balance, boolean merchant) {
        this.handle = handle;
        this.displayName = displayName;
        this.balance = balance;
        this.merchant = merchant;
        this.openedAt = Instant.now();
    }

    public String getHandle() { return handle; }
    public void setHandle(String v) { this.handle = v; }

    public String getDisplayName() { return displayName; }
    public void setDisplayName(String v) { this.displayName = v; }

    public Double getBalance() { return balance; }
    public void setBalance(Double v) { this.balance = v; }

    public Boolean getMerchant() { return merchant; }
    public void setMerchant(Boolean v) { this.merchant = v; }

    public Instant getOpenedAt() { return openedAt; }
    public void setOpenedAt(Instant v) { this.openedAt = v; }
}

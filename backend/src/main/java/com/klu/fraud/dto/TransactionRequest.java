package com.klu.fraud.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.PositiveOrZero;

/**
 * What the dashboard (or any caller) posts to submit a payment.
 *
 * <p>Validated with annotations so a bad request is rejected before it reaches
 * the database or the ML service. Silently defaulting a missing amount to zero
 * would produce a confident decision computed from nothing.
 */
public class TransactionRequest {

    @NotBlank(message = "type is required (TRANSFER, CASH_OUT, PAYMENT, CASH_IN or DEBIT)")
    private String type;

    @NotNull(message = "amount is required")
    @PositiveOrZero(message = "amount cannot be negative")
    private Double amount;

    @NotBlank(message = "nameOrig (the sending account) is required")
    private String nameOrig;

    @NotNull @PositiveOrZero
    private Double oldbalanceOrg;

    @NotNull @PositiveOrZero
    private Double newbalanceOrig;

    @NotBlank(message = "nameDest (the receiving account) is required")
    private String nameDest;

    @NotNull @PositiveOrZero
    private Double oldbalanceDest;

    @NotNull @PositiveOrZero
    private Double newbalanceDest;

    // Optional
    private Integer step;
    private String location;
    private Double locationLat;
    private Double locationLon;
    private Boolean trustedDevice = false;

    public String getType() { return type; }
    public void setType(String v) { this.type = v; }

    public Double getAmount() { return amount; }
    public void setAmount(Double v) { this.amount = v; }

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

    public Integer getStep() { return step; }
    public void setStep(Integer v) { this.step = v; }

    public String getLocation() { return location; }
    public void setLocation(String v) { this.location = v; }

    public Double getLocationLat() { return locationLat; }
    public void setLocationLat(Double v) { this.locationLat = v; }

    public Double getLocationLon() { return locationLon; }
    public void setLocationLon(Double v) { this.locationLon = v; }

    public Boolean getTrustedDevice() { return trustedDevice; }
    public void setTrustedDevice(Boolean v) { this.trustedDevice = v; }
}

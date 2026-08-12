package com.klu.fraud;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.scheduling.annotation.EnableScheduling;

/**
 * Explainable Real-Time Fraud Detection - Spring Boot backend.
 *
 * <p>Scheduling is enabled because OTP challenges must expire on their own.
 * Without it an unanswered payment would sit undecided forever, which would
 * quietly reintroduce the human step this system exists to remove.
 */
@SpringBootApplication
@EnableScheduling
public class FraudDetectionApplication {

    public static void main(String[] args) {
        SpringApplication.run(FraudDetectionApplication.class, args);
        System.out.println("""

                ======================================================================
                  Fraud detection backend is up:  http://localhost:8080
                ======================================================================
                  Dashboard        http://localhost:8080/
                  H2 console       http://localhost:8080/h2-console
                  Health           http://localhost:8080/api/health
                  Submit a payment POST http://localhost:8080/api/transactions

                  Every decision is automatic. There is no human reviewer.
                ======================================================================
                """);
    }
}

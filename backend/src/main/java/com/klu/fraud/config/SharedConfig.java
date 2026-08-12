package com.klu.fraud.config;

import jakarta.annotation.PostConstruct;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;
import org.yaml.snakeyaml.Yaml;

/**
 * Reads the SHARED config.yaml at the project root - the same file the Python
 * service reads.
 *
 * <p>Why not just put these numbers in application.properties? Because then the
 * approve/block thresholds would exist in two files, in two languages, and the
 * day someone tunes one and forgets the other, Python and Java would disagree
 * about what counts as fraud. The bug would be silent: both services keep
 * answering, they just stop agreeing. One file, one source of truth.
 *
 * <p>If config.yaml cannot be found we fall back to built-in defaults and log
 * loudly, rather than refusing to start - a demo that boots with a warning
 * beats a demo that will not boot at all.
 */
@Component
public class SharedConfig {

    private static final Logger log = LoggerFactory.getLogger(SharedConfig.class);

    // Decision thresholds
    private double approveBelow = 0.35;
    private double blockAbove = 0.75;
    private int otpTimeoutSeconds = 60;

    // Where the Python service lives, and how long we wait for it
    private String mlServiceUrl = "http://localhost:5001";
    private int mlTimeoutSeconds = 3;

    // Rule-layer settings, mirrored from Python so both sides agree
    private int rapidSenderWindowSteps = 1;
    private int rapidSenderMaxAllowed = 3;
    private int rapidReceiverWindowSteps = 1;
    private int rapidReceiverMaxAllowed = 4;

    private boolean loadedFromFile = false;
    private String configPath = "(not found)";

    @PostConstruct
    @SuppressWarnings("unchecked")
    public void load() {
        Path found = findConfigFile();
        if (found == null) {
            log.warn("=================================================================");
            log.warn("config.yaml NOT FOUND. Using built-in defaults.");
            log.warn("Java and Python may now disagree about the thresholds.");
            log.warn("Expected it at the project root, next to the ml-service folder.");
            log.warn("=================================================================");
            return;
        }

        try (InputStream in = Files.newInputStream(found)) {
            Map<String, Object> root = new Yaml().load(in);

            Map<String, Object> decision = (Map<String, Object>) root.get("decision");
            if (decision != null) {
                approveBelow = toDouble(decision.get("approve_below"), approveBelow);
                blockAbove = toDouble(decision.get("block_above"), blockAbove);
                otpTimeoutSeconds = toInt(decision.get("otp_timeout_seconds"),
                        otpTimeoutSeconds);
            }

            Map<String, Object> services = (Map<String, Object>) root.get("services");
            if (services != null) {
                Map<String, Object> backend = (Map<String, Object>) services.get("backend");
                if (backend != null) {
                    mlServiceUrl = String.valueOf(
                            backend.getOrDefault("ml_service_url", mlServiceUrl));
                    mlTimeoutSeconds = toInt(backend.get("ml_timeout_seconds"),
                            mlTimeoutSeconds);
                }
            }

            Map<String, Object> rules = (Map<String, Object>) root.get("rules");
            if (rules != null) {
                Map<String, Object> sender =
                        (Map<String, Object>) rules.get("rapid_sender_transfers");
                if (sender != null) {
                    rapidSenderWindowSteps = toInt(sender.get("window_steps"),
                            rapidSenderWindowSteps);
                    rapidSenderMaxAllowed = toInt(sender.get("max_allowed"),
                            rapidSenderMaxAllowed);
                }
                Map<String, Object> receiver =
                        (Map<String, Object>) rules.get("rapid_receiver_collection");
                if (receiver != null) {
                    rapidReceiverWindowSteps = toInt(receiver.get("window_steps"),
                            rapidReceiverWindowSteps);
                    rapidReceiverMaxAllowed = toInt(receiver.get("max_allowed"),
                            rapidReceiverMaxAllowed);
                }
            }

            loadedFromFile = true;
            configPath = found.toAbsolutePath().toString();
            log.info("Loaded shared config from {}", configPath);
            log.info("  APPROVE below {}, BLOCK above {}, OTP timeout {}s",
                    approveBelow, blockAbove, otpTimeoutSeconds);
            log.info("  ML service at {} (timeout {}s)", mlServiceUrl, mlTimeoutSeconds);

        } catch (Exception e) {
            log.error("Could not parse config.yaml ({}). Using defaults.",
                    e.getMessage());
        }
    }

    /** Walk up from the working directory looking for config.yaml. */
    private Path findConfigFile() {
        List<String> candidates = List.of(
                "../config.yaml", "config.yaml", "../../config.yaml");
        for (String candidate : candidates) {
            Path path = Paths.get(candidate);
            if (Files.exists(path)) {
                return path;
            }
        }
        return null;
    }

    private static double toDouble(Object value, double fallback) {
        return value instanceof Number n ? n.doubleValue() : fallback;
    }

    private static int toInt(Object value, int fallback) {
        return value instanceof Number n ? n.intValue() : fallback;
    }

    public double getApproveBelow() { return approveBelow; }
    public double getBlockAbove() { return blockAbove; }
    public int getOtpTimeoutSeconds() { return otpTimeoutSeconds; }
    public String getMlServiceUrl() { return mlServiceUrl; }
    public int getMlTimeoutSeconds() { return mlTimeoutSeconds; }
    public int getRapidSenderWindowSteps() { return rapidSenderWindowSteps; }
    public int getRapidSenderMaxAllowed() { return rapidSenderMaxAllowed; }
    public int getRapidReceiverWindowSteps() { return rapidReceiverWindowSteps; }
    public int getRapidReceiverMaxAllowed() { return rapidReceiverMaxAllowed; }
    public boolean isLoadedFromFile() { return loadedFromFile; }
    public String getConfigPath() { return configPath; }
}

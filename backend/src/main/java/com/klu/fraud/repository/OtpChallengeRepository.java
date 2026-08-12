package com.klu.fraud.repository;

import com.klu.fraud.model.OtpChallenge;
import com.klu.fraud.model.OtpStatus;
import java.util.List;
import java.util.Optional;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface OtpChallengeRepository extends JpaRepository<OtpChallenge, Long> {

    Optional<OtpChallenge> findByTransactionIdAndStatus(Long transactionId,
                                                        OtpStatus status);

    List<OtpChallenge> findByStatus(OtpStatus status);

    List<OtpChallenge> findAllByOrderBySentAtDesc();
}

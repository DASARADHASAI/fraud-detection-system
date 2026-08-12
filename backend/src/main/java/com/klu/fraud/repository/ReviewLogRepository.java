package com.klu.fraud.repository;

import com.klu.fraud.model.ReviewLog;
import java.util.List;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface ReviewLogRepository extends JpaRepository<ReviewLog, Long> {

    List<ReviewLog> findByTransactionIdOrderByLoggedAtAsc(Long transactionId);

    List<ReviewLog> findAllByOrderByLoggedAtDesc(Pageable pageable);
}

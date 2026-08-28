package com.klu.fraud.repository;

import com.klu.fraud.model.Decision;
import com.klu.fraud.model.Transaction;
import java.time.Instant;
import java.util.List;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

@Repository
public interface TransactionRepository extends JpaRepository<Transaction, Long> {

    List<Transaction> findAllByOrderByReceivedAtDesc(Pageable pageable);

    List<Transaction> findByFinalDecisionOrderByReceivedAtDesc(Decision decision);

    /** Everything that was not simply approved - what the dashboard highlights. */
    @Query("SELECT t FROM Transaction t WHERE t.finalDecision <> 'APPROVE' "
            + "ORDER BY t.receivedAt DESC")
    List<Transaction> findAlerts(Pageable pageable);

    /**
     * Has this customer successfully paid this receiver before?
     * Used by the quick-signal check: a payee you have paid before and which
     * went through is far less likely to be a fraudster's account.
     */
    @Query("SELECT COUNT(t) FROM Transaction t WHERE t.nameOrig = :orig "
            + "AND t.nameDest = :dest AND t.finalDecision = 'APPROVE' AND t.id <> :excludeId")
    long countPreviousApprovedPayments(@Param("orig") String orig,
                                       @Param("dest") String dest,
                                       @Param("excludeId") Long excludeId);

    /** How many payments this account has SENT since a given moment. */
    @Query("SELECT COUNT(t) FROM Transaction t WHERE t.nameOrig = :account "
            + "AND t.receivedAt >= :since AND t.id <> :excludeId")
    long countSentSince(@Param("account") String account,
                        @Param("since") Instant since,
                        @Param("excludeId") Long excludeId);

    /** How many payments this account has RECEIVED since a given moment. */
    @Query("SELECT COUNT(t) FROM Transaction t WHERE t.nameDest = :account "
            + "AND t.receivedAt >= :since AND t.id <> :excludeId")
    long countReceivedSince(@Param("account") String account,
                            @Param("since") Instant since,
                            @Param("excludeId") Long excludeId);

    long countByFinalDecision(Decision decision);

    /**
     * The payees this customer has actually paid, successfully, before.
     *
     * <p>This replaces UniPay's hard-coded contact list. A payee is on it
     * because a payment to them went through - not because the page said so.
     * On a fresh account it is empty, which is correct: the first payment to
     * anybody IS a payment to a new receiver, and that is a genuine signal the
     * quick-signal layer reads.
     *
     * <p>Returns rows of [handle, paymentCount, lastPaidAt].
     */
    @Query("SELECT t.nameDest, COUNT(t), MAX(t.receivedAt) FROM Transaction t "
            + "WHERE t.nameOrig = :orig AND t.finalDecision = 'APPROVE' "
            + "GROUP BY t.nameDest ORDER BY MAX(t.receivedAt) DESC")
    List<Object[]> findPayeesOf(@Param("orig") String orig);
}

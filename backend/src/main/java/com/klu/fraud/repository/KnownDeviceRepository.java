package com.klu.fraud.repository;

import com.klu.fraud.model.KnownDevice;
import java.util.List;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface KnownDeviceRepository extends JpaRepository<KnownDevice, String> {

    List<KnownDevice> findByAccountHandle(String accountHandle);
}

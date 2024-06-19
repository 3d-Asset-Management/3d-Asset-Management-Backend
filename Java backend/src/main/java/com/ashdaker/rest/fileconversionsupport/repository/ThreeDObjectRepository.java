package com.ashdaker.rest.fileconversionsupport.repository;


import com.ashdaker.rest.fileconversionsupport.entity.ThreeDObject;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface ThreeDObjectRepository extends JpaRepository<ThreeDObject, Long> {
}

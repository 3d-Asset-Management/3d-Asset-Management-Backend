package com.ashdaker.rest.fileconversionsupport.service;

import com.ashdaker.rest.fileconversionsupport.entity.ThreeDObject;
import com.ashdaker.rest.fileconversionsupport.repository.ThreeDObjectRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class ThreeDObjectService {
    private final ThreeDObjectRepository repository;

    @Autowired
    public ThreeDObjectService(ThreeDObjectRepository repository) {
        this.repository = repository;
    }

    public List<ThreeDObject> getAllThreeDObjects() {
        return repository.findAll();
    }

    public ThreeDObject getThreeDObjectById(Long id) {
        return repository.findById(id).orElse(null);
    }

    public ThreeDObject saveThreeDObject(ThreeDObject threeDObject) {
        return repository.save(threeDObject);
    }

    public void deleteThreeDObject(Long id) {
        repository.deleteById(id);
    }

    public String getFilePathById(Long id) {
        ThreeDObject threeDObject = repository.findById(id).orElse(null);
        return threeDObject != null ? threeDObject.getFilePath() : null;
    }
}

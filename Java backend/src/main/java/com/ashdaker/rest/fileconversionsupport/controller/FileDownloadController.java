package com.ashdaker.rest.fileconversionsupport.controller;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.List;

import com.ashdaker.rest.fileconversionsupport.entity.ThreeDObject;
import com.ashdaker.rest.fileconversionsupport.service.ThreeDObjectService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.InputStreamResource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/3dobjects")
public class FileDownloadController
{
    private final ThreeDObjectService service;

    @Autowired
    public FileDownloadController(ThreeDObjectService service) {
        this.service = service;
    }

    @GetMapping
    public List<ThreeDObject> getAllThreeDObjects() {
        return service.getAllThreeDObjects();
    }

    @GetMapping("/{id}")
    public ResponseEntity<ThreeDObject> getThreeDObjectById(@PathVariable Long id) {
        ThreeDObject threeDObject = service.getThreeDObjectById(id);
        if (threeDObject == null) {
            return ResponseEntity.notFound().build();
        }
        return ResponseEntity.ok(threeDObject);
    }

    @PostMapping
    public ThreeDObject createThreeDObject(@RequestBody ThreeDObject threeDObject) {
        return service.saveThreeDObject(threeDObject);
    }

    @PutMapping("/{id}")
    public ResponseEntity<ThreeDObject> updateThreeDObject(@PathVariable Long id, @RequestBody ThreeDObject updatedThreeDObject) {
        ThreeDObject existingThreeDObject = service.getThreeDObjectById(id);
        if (existingThreeDObject == null) {
            return ResponseEntity.notFound().build();
        }
        updatedThreeDObject.setId(id);
        return ResponseEntity.ok(service.saveThreeDObject(updatedThreeDObject));
    }

    @DeleteMapping("/{id}")
    public ResponseEntity<Void> deleteThreeDObject(@PathVariable Long id) {
        if (service.getThreeDObjectById(id) == null) {
            return ResponseEntity.notFound().build();
        }
        service.deleteThreeDObject(id);
        return ResponseEntity.noContent().build();
    }

//    @GetMapping
    @RequestMapping(value = "/download/{id}", method = RequestMethod.GET)
    public ResponseEntity<Object> downloadFile(@PathVariable Long id) throws IOException
    {
        String filename = service.getFilePathById(id);
        if (filename == null) {
            return ResponseEntity.notFound().build();
        }
//        String filename = "/Users/pdhaker/Downloads/tree.jpg";
        System.out.println(filename);
        File file = new File(filename);
        InputStreamResource resource = new InputStreamResource(new FileInputStream(file));

        HttpHeaders headers = new HttpHeaders();
        headers.add("Content-Disposition",
                String.format("attachment; filename=\"%s\"", file.getName()));
        headers.add("Cache-Control", "no-cache, no-store, must-revalidate");
        headers.add("Pragma", "no-cache");
        headers.add("Expires", "0");

        ResponseEntity<Object> responseEntity = ResponseEntity.ok().headers(headers)
                .contentLength(file.length())
                .contentType(MediaType.parseMediaType("application/txt")).body(resource);

        return responseEntity;
    }
}

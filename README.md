# Project for detecting and classify Vulnerable Road Users (VRU) on road.
## Image Processing Utilities
### Crop Images
#### *Cropping out objects and reshape them into the suitable size for the neural network*
1. Modify global variables `RAW_DATA_PATH`, `PROCESSED_TRAIN_DATA_PATH`, `PROCESSED_VAL_DATA_PATH`, `JSON_F`, `NEW_SHAPE` and `TRAIN_TO_VAL_RATIO` in file `process_img_from_json.py`
2. Run `python process_img_from_json.py`
### Detect Objects in Images
#### *Detect objects with YOLO v3 object detection model*
1. Put images to be processed in `<IMAGE_DIR>`
2. Run `python obj_det/det.py --images <IMAGE_DIR> --det <OUTPUT_DIR>`
### Merge Bounding Boxes
#### *Find bounding box groups that are potentially wheelchair users and merge the bounding boxes*
1. Run `python test.py`

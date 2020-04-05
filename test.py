from __future__ import print_function
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.optim as optim
import json
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score
from VRUDataset import VRUDataset
from models.cnn import CNN

# from obj_det.darknet import Darknet
# from obj_det.util import load_classes

import copy
import pickle
import argparse

JSON_F = "obj_det/detected.json"
POTENTIAL_LIST = ["chair", "bench", "bicycle", "motorbike"]
NEW_PERSON_THRESH = 0.9
OVERLAP_PERC = 0.5



# feed image into classification algorithm and print result
def classify_img():
    pass

# crop and save image defined by all bbox in given json
def crop_image_with_processed_bbox(img, j):
    pass


# if a bbox labelled person overlaps by OVERLAP_PERC
# with at least one labelled bbox in the following:
# motorbike, bicycle, chair, bench ...
# then append a large bbox including both to json
# return: list of processed bbox
def process_bbox(img, j):
    processed = []
    persons = []
    person_range = set()
    for obj in j:
        if obj["label"] == "person":
            px = set(range(obj["left"], obj["right"]))
            py = set(range(obj["up"], obj["down"]))
            this_person = set((x, y) for x in px for y in py)
            # if (float(len(this_person & person_range)) / float(len(this_person)) < NEW_PERSON_THRESH):
            person_range |= this_person
            persons.append(this_person)
    for obj in j:
        if obj["label"] in POTENTIAL_LIST:
            def overlap(persons, obj):
                ox = set(range(obj["left"], obj["right"]))
                oy = set(range(obj["up"], obj["down"]))
                this_obj = set((x, y) for x in ox for y in oy)
                if (float(len(this_obj & person_range)) / float(len(this_obj)) > OVERLAP_PERC):
                    for i, p in enumerate(persons):
                        if (float(len(this_obj & p)) / float(len(this_obj)) > OVERLAP_PERC):
                            processed.append({
                                "label_debug": obj["label"],
                                "up": 1,
                                "down": 2,
                                "left": 3,
                                "right": 4
                            })
            overlap(persons, obj)
    blah = persons[0]
    for i in persons:
        blah &= i
    print(blah)
    print("should be zero: ", len(blah))
    print(processed)
    print(len(processed))
    return processed
    pass

def run_obj_det():
    os.system("\
                cd obj_det && \
                rm det/* || true && \
                python det.py --images imgs --det det && \
                cd ..");

# interate list of image
def run():
    # run_obj_det()
    # something
    js = {}
    with open(JSON_F, 'r') as f:
        js = json.load(f)
    new_js = {}
    for j in js:
        new_js[j] = process_bbox(j, js[j])
    pass

if __name__ == "__main__":
    run()
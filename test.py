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
import argparse
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
    person_obj = []
    person_range = set()
    for obj in j:
        if obj["label"] == "person":
            px = set(range(obj["left"], obj["right"]))
            py = set(range(obj["up"], obj["down"]))
            this_person = set((x, y) for x in px for y in py)
            # if (float(len(this_person & person_range)) / float(len(this_person)) < NEW_PERSON_THRESH):
            person_range |= this_person
            persons.append(this_person)
            person_obj.append(obj)
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
                                "up": min(obj["up"], person_obj[i]["up"]),
                                "down": max(obj["down"], person_obj[i]["down"]),
                                "left": min(obj["left"], person_obj[i]["left"]),
                                "right": max(obj["right"], person_obj[i]["right"])
                            })
                            persons.pop(i)
                            person_obj.pop(i)
                            break;
            overlap(persons, obj)
            
    # print(processed)
    # print(len(processed))
    return processed
    pass

def run_obj_det():
    os.system("\
                cd obj_det && \
                rm det/* || true && \
                python det.py --images imgs --det det && \
                cd ..");

def save_img_with_new_bbox(new_js):
    for img_name in new_js:
        img = cv2.imread(os.path.join(os.getcwd(), "obj_det/imgs/", img_name))
        cv2.imshow(img_name, img)
        for obj in new_js[img_name]:
            print("an object!")
            c1 = tuple((obj["left"], obj["up"]))
            c2 = tuple((obj["right"], obj["down"]))
            import pickle as pkl
            import random
            color = random.choice(pkl.load(open("obj_det/pallete", "rb")))
            cv2.rectangle(img, c1, c2, color, 2)
        cv2.imwrite("new_bbox/new_" + img_name, img)

# interate list of image
def run():
    # run_obj_det()
    js = {}
    with open(JSON_F, 'r') as f:
        js = json.load(f)
    new_js = {}
    for j in js:
        new_js[j] = process_bbox(j, js[j])
    return new_js

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', action='store_true', help="save img w/ new bbox")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    new_js = run()
    print(new_js)
    if args.b:
        save_img_with_new_bbox(new_js)
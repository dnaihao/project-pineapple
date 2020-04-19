from __future__ import print_function
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.optim as optim
import torchvision.models as models
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
from models.autoencoder import Autoencoder
from process_img_from_json import crop_and_resize_image, num_objects, find_region

# from obj_det.darknet import Darknet
# from obj_det.util import load_classes

import copy
import pickle
import argparse
import glob

JSON_F = "obj_det/detected.json"
POTENTIAL_LIST = ["chair", "bench", "bicycle", "motorbike"]
NEW_PERSON_THRESH = 0.9
OVERLAP_PERC = 0.5
CLASSIFIER_PATH = os.path.join(".", "Saved Models", "model")


def classify_img(img, epoch_to_load=-1):
    # Given an input image, classify the image with the classifier
    # Input- img: H x W x Z Numpy Tensor (channel first)
    #        epoch_to_load: which epoch's model/params to load (idx
    #                       starting from 0), default loading the 
    #                       last epoch's model/params
    # 
    # Output- : label associated with the feeded image

    map_location = None
    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'

    saved_models = glob.glob(os.path.join(CLASSIFIER_PATH, "*.pt"))
    # Assumed model saved in a pattern with larger ones corresponding to
    # later epochs
    saved_models.sort()

    model_to_load = saved_models[epoch_to_load]
    model = torch.load(model_to_load, map_location=map_location)

    ######################################################
    ######################################################
    # TODO: interprete tensor as whether the img has 
    # wheelchair user
    # example of the current output:
    #   tensor([[ 0.9059, -0.4302]], grad_fn=<AddmmBackward>)
    ######################################################
    ######################################################

    return model(img)


def crop_image_with_processed_bbox(img_name, j):
    # Given an input image, crop image by the information specified in json file
    # Input- img_name: name of the image
    #        j: json file specifies the information for the bounding box
    # 
    # Output- : matrix represent the bounding box
    pass


# Determines if CNN output is wheelchair or not
# Input - torch tensor
# Example input - 
#       tensor([[ 0.9059, -0.4302]], grad_fn=<AddmmBackward>)
# output - false
def is_wheelchair(output): 
    if output[0][0]>output[0][1]:
        return False
    else:
        return True
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
    with open("new_bbox.json", 'w') as f:
        json.dump(new_js, f)
    return new_js

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', action='store_true', help="save img w/ new bbox")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    new_js = run()
    if args.b:
        save_img_with_new_bbox(new_js)
    
    ################################################
    ################################################
    ######### Testing for second layer #############
    ################################################
    ################################################
    for img_name in new_js.keys():
        for idx in range(num_objects(img_name, new_js)):
            region = find_region(img_name, new_js, idx)
            cropped_img = crop_and_resize_image(img_name, region, idx, train=False, save=False)

            # resize the cropped image
            new_size = (100, 100)
            # resize the cropped image
            import PIL
            assert type(cropped_img) is PIL.Image.Image
            cropped_img = cropped_img.resize(new_size)

            # Convert from PIL image to numpy array
            cropped_img = np.array(cropped_img)

            ######################################################
            ######################################################
            # TODO: deal with the 4 dimensions in PNG 
            # right now the second layer model can only
            # handle 3 dimension JPG pics
            ######################################################
            ######################################################
            
            # Convert from channel last to channel first np array
            cropped_img = np.moveaxis(cropped_img, -1, 0)
            # Expand the dimension for it to fit the model weight
            cropped_img = np.expand_dims(cropped_img, axis=0)
            label = classify_img(torch.Tensor(cropped_img))
            output  = is_wheelchair(label)

            print(label)
            print(output)

    ################################################
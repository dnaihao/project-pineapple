import json
import os
from PIL import Image
import argparse

RAW_DATA_PATH = os.path.join("obj_det", "imgs")
PROCESSED_TRAIN_DATA_PATH = "C:\\Users\\shubh\\OneDrive\\Desktop\\ENTR 390\\Dataset\\Train"
PROCESSED_VAL_DATA_PATH = os.path.join("obj_det", "imgs")
JSON_F = "C:\\Users\\shubh\\OneDrive\\Desktop\\ENTR 390\\Dataset\\Mobility.json"
NEW_SHAPE = (100, 100)
TRAIN_TO_VAL_RATIO = 0.7


def gen_json():
    json_f = {
        'obj': [
            {
                'f_name' : 'lol.png',
                'label': 'dude',
                'size_x': 540,
                'size_y': 900,
                'dim': [180,30,180,30] # xmax, xmin, ymax, ymin
            }, 
            {
                'f_name' : 'ksk.png',
                'label': 'eecs',
                'size_x': 540,
                'size_y': 900,
                'dim': [180,30,180,30] # xmax, xmin, ymax, ymin
            }, 
        ]
    }
    return json_f

def write_json(j, train=True):
    if train:
        print("write train json file...")
        with open('train.json', 'w') as f:
            json.dump(j, f)
    else:
        print("write val json file...")
        with open('val.json', 'w') as f:
            json.dump(j, f)

# process images and create updated json file
def process_images():
    train_j = {'obj':[]}
    val_j = {'obj':[]}
    j = {}
    with open(JSON_F, 'r') as f:
        j = json.load(f)
    index = 0
    divide = len(j['obj']) * TRAIN_TO_VAL_RATIO
    for idx, obj in enumerate(j['obj']):
        if (idx < divide):
            if obj['label'] != 'NA':
                region = (obj['dim'][1], obj['dim'][3], obj['dim'][0], obj['dim'][2])
                f_name = crop_and_resize_image(obj['f_name'], region=region, index=index, train=True)
                obj['f_name'] = f_name
                if obj['label']=='wheelchair':
                    train_j['obj'].append({
                        'f_name': f_name,
                        'label': obj['label'],
                        'size_x': obj['size_x'],
                        'size_y': obj['size_y'],
                        'dim': region
                        })
                    index += 1
                else:
                    train_j['obj'].append({
                        'f_name': f_name,
                        'label': 'not_wheelchair',
                        'size_x': obj['size_x'],
                        'size_y': obj['size_y'],
                        'dim': region
                        })
                    index += 1
            if (idx % 100 == 0):
                write_json(train_j, train=True)
        else:
            if obj['label'] != 'NA':
                region = (obj['dim'][1], obj['dim'][3], obj['dim'][0], obj['dim'][2])
                f_name = crop_and_resize_image(obj['f_name'], region=region, index=index, train=False)
                obj['f_name'] = f_name
                if obj['label']=='wheelchair':
                    val_j['obj'].append({
                        'f_name': f_name,
                        'label': obj['label'],
                        'size_x': obj['size_x'],
                        'size_y': obj['size_y'],
                        'dim': region
                        })
                    index += 1
                else:
                    val_j['obj'].append({
                        'f_name': f_name,
                        'label': 'not_wheelchair',
                        'size_x': obj['size_x'],
                        'size_y': obj['size_y'],
                        'dim': region
                        })
                    index += 1
            if (idx % 100 == 0):
                write_json(val_j, train=False)

def num_objects(f_name, j):
    # Given json file and the image file name, return number of objects detected
    # in the image.
    #
    # Input- f_name: name of the image (string)
    #        j: json file (Dict)
    # 
    # Output- : number of objects detected in the image
    return len(j.get(f_name))


def find_region(f_name, j, idx):
    # Given json file and the image file name and the index of the object
    # , return the region of the object in the json file.
    #
    # Input- f_name: name of the image (string)
    #        j: json file (Dict)
    #        idx: index of the object (Int)
    # 
    # Output- : tuple represents the region of the object
    try:
        obj = j.get(f_name)[idx]
        return (obj.get("left"), obj.get("up"), obj.get("right"), obj.get("down"))
    except Exception as e:
        print("Exception happened in finding region: {}".format(str(e)))
    
      
def crop_and_resize_image(f_name, region, index, train=True, save=True):
    img_path = os.path.join(os.getcwd(), RAW_DATA_PATH, f_name)
    if f_name.endswith('.jpg') or f_name.endswith('.png'):
        f_name = f_name[:-4] + '_' + str(index) + f_name[-4:]
    if train:
        save_path = os.path.join(os.getcwd(), PROCESSED_TRAIN_DATA_PATH, f_name)
    else:
        save_path = os.path.join(os.getcwd(), PROCESSED_VAL_DATA_PATH, f_name)
    try:
        img = Image.open(img_path)
        img = img.crop(region).resize(NEW_SHAPE)
        if save:
            img.save(save_path)
            return f_name
    except:
        print("failed to process image")
        exit(1)
    return img

def parse_args():
    global RAW_DATA_PATH, PROCESSED_TRAIN_DATA_PATH, PROCESSED_VAL_DATA_PATH, JSON_F, NEW_SHAPE
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--shape", nargs=2, default=(32, 32), required=False)
    parser.add_argument("-j", "--json", nargs='*', default="my.json", required=False)
    parser.add_argument("-r", "--raw", nargs=1, default="d", required=False)
    parser.add_argument("-p", "--processed", nargs=1, default="done", required=False)
    args = parser.parse_args()
    # RAW_DATA_PATH = args.raw
    # PROCESSED_DATA_PATH = args.processed
    # JSON_F = args.json
    # NEW_SHAPE = (int(args.shape[0]), int(args.shape[1]))

if __name__ == "__main__":
    ans = "pineapple"
    if os.path.exists("val.json") or os.path.exists("val.json"):
        ans = input("YOU SURE??? ")
    if (ans != "pineapple"):
        exit()
    parse_args()
    process_images()

        

    

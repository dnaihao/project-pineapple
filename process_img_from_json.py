import json
import os
from PIL import Image
import argparse

RAW_DATA_PATH = "d"
PROCESSED_DATA_PATH = "done"
JSON_F = "my.json"
NEW_SHAPE = (32, 32)


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

def write_json(j):
    print("write json file...")
    with open('new.json', 'w') as f:
        json.dump(j, f)

# process images and create updated json file
def process_images():
    j = {}
    with open(JSON_F, 'r') as f:
        j = json.load(f)
    index = 0
    for obj in j['obj']:
        region = (obj['dim'][1], obj['dim'][3], obj['dim'][0], obj['dim'][2])
        f_name = crop_and_resize_image(obj['f_name'], region=region, index=index)
        obj['f_name'] = f_name
        index += 1
    write_json(j)
        
def crop_and_resize_image(f_name, region, index):
    img_path = os.path.join(os.getcwd(), RAW_DATA_PATH, f_name)
    if f_name.endswith('.jpg') or f_name.endswith('.png'):
        f_name = f_name[:-4] + '_' + str(index) + f_name[-4:]

    save_path = os.path.join(os.getcwd(), PROCESSED_DATA_PATH, f_name)
    try:
        img = Image.open(img_path)
        img = img.crop(region).resize(NEW_SHAPE)
        img.save(save_path)
    except:
        print("failed to process image")
        exit(1)
    return f_name

def parse_args():
    global RAW_DATA_PATH, PROCESSED_DATA_PATH, JSON_F, NEW_SHAPE
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--shape", nargs=2, default=(32, 32), required=False)
    parser.add_argument("-j", "--json", nargs=1, default="my.json", required=False)
    parser.add_argument("-r", "--raw", nargs=1, default="d", required=False)
    parser.add_argument("-p", "--processed", nargs=1, default="done", required=False)
    args = parser.parse_args()
    RAW_DATA_PATH = args.raw
    PROCESSED_DATA_PATH = args.processed
    JSON_F = args.json
    NEW_SHAPE = (int(args.shape[0]), int(args.shape[1]))

if __name__ == "__main__":
    parse_args()
    process_images()

        

    

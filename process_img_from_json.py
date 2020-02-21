import json
import os
from PIL import Image
import argparse

RAW_DATA_PATH = "C:\\Users\\shubh\\OneDrive\\Desktop\\ENTR 390\\Dataset\\Images_RGB"
PROCESSED_DATA_PATH = "C:\\Users\\shubh\\OneDrive\\Desktop\\ENTR 390\\Dataset\\Im"
JSON_F = "C:\\Users\\shubh\\OneDrive\\Desktop\\ENTR 390\\Dataset\\Mobility.json"
NEW_SHAPE = (100, 100)


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
    new_j = {'obj':[]}
    j = {}
    with open(JSON_F, 'r') as f:
        j = json.load(f)
    index = 0
    print(j)
    for idx, obj in enumerate(j['obj']):
        if obj['label'] != 'NA':
            region = (obj['dim'][1], obj['dim'][3], obj['dim'][0], obj['dim'][2])
            f_name = crop_and_resize_image(obj['f_name'], region=region, index=index)
            obj['f_name'] = f_name
            new_j['obj'].append({
                'f_name': f_name,
                'label': obj['label'],
                'size_x': obj['size_x'],
                'size_y': obj['size_y'],
                'dim': region
                })
            index += 1
        if (idx % 100 == 0):
            write_json(new_j)
    write_json(new_j)
        
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
    parser.add_argument("-j", "--json", nargs='*', default="my.json", required=False)
    parser.add_argument("-r", "--raw", nargs=1, default="d", required=False)
    parser.add_argument("-p", "--processed", nargs=1, default="done", required=False)
    args = parser.parse_args()
    # RAW_DATA_PATH = args.raw
    # PROCESSED_DATA_PATH = args.processed
    # JSON_F = args.json
    # NEW_SHAPE = (int(args.shape[0]), int(args.shape[1]))

if __name__ == "__main__":
    parse_args()
    process_images()

        

    

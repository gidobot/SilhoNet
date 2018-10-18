import json
import os
import sys
from scipy.io import loadmat
import argparse
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Convert YCB annotations to COCO json format')
    parser.add_argument(
        '--outdir', help="output dir for json files", default=None, type=str)
    parser.add_argument(
        '--datadir', help="root directory of YCB video dataset",
        default=None, type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def read_categories(f):
    class_dict = {}
    with open(f) as fd:
        lines = fd.readlines()
    lines = [x.strip() for x in lines if x.strip()]
    for class_id, class_name in enumerate(lines):
        class_dict[class_name] = class_id + 1
    return class_dict 


def read_image_set(f):
    with open(f) as fd:
        lines = fd.readlines()
    lines = [x.strip() for x in lines if x.strip()]
    return lines


def read_bbox(f):
    with open(f) as fd:
        roi_line = fd.readlines()
    roi_line = [x.strip() for x in roi_line if x.strip()]
    bbox_dict = {}
    for line in roi_line:
        roi_list = line.split()
        # roi coordinates (y1, x1, y2, x2)
        roi_coords = [roi_list[2], roi_list[1], roi_list[4], roi_list[3]]
        roi_coords = [float(x) for x in roi_coords]
        width = roi_coords[3] - roi_coords[1]
        height = roi_coords[2] - roi_coords[0]
        if width < 0 or height < 0:
            continue
        area = int(width*height)
        bbox = [roi_coords[1], roi_coords[0], width, height]
        bbox = [int(x) for x in bbox]
        bbox_dict[roi_list[0]] = {}
        bbox_dict[roi_list[0]]['bbox'] = bbox
        bbox_dict[roi_list[0]]['area'] = area
    return bbox_dict


def convert_ycb_annotations(data_dir, out_dir):
    sets = ['small', 'train', 'val', 'keyframe', 'synthetic', 'trainsyn']
    for image_set in sets:
        ann_dict = {}
        images = []
        annotations = []
        class_name_file = "classes.txt"
        json_name = 'ycb_video_%s.json'
        class_dict = read_categories(class_name_file)
        ann_id = 0
        image_id = 0
        image_list = read_image_set(os.path.join(data_dir, "image_sets", image_set + ".txt"))
        pbar = tqdm(desc=image_set, total=len(image_list))
        for image_name in image_list:
            image = {}
            image['id'] = image_id
            image_id += 1
            image['width'] = 640
            image['height'] = 480
            image['file_name'] = image_name + "-color.png"
            image['seg_file_name'] = image_name + "-label.png"
            images.append(image)

            bbox_dict = read_bbox(os.path.join(data_dir, "data", image_name + '-box.txt'))
            for key in bbox_dict:
                ann = {}
                ann['id'] = ann_id
                ann_id += 1
                ann['image_id'] = image['id']
                ann['category_id'] = class_dict[key]
                ann['iscrowd'] = 0
                ann['area'] = bbox_dict[key]['area']
                ann['bbox'] = bbox_dict[key]['bbox']
                ann['segmentation'] = []
                annotations.append(ann)
            pbar.update(1)
        pbar.close()
        ann_dict['images'] = images
        categories = [{"id": class_dict[name], "name": name} for name in class_dict]
        ann_dict['categories'] = categories
        ann_dict['annotations'] = annotations
        print("Num categories: %s" % len(categories))
        print("Num images: %s" % len(images))
        print("Num annotations: %s" % len(annotations))
        with open(os.path.join(out_dir, json_name % image_set), 'wb') as outfile:
            outfile.write(json.dumps(ann_dict))


if __name__ == '__main__':
    args = parse_args()
    convert_ycb_annotations(args.datadir, args.outdir)

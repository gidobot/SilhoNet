import json
import os
import sys
from scipy.io import loadmat
import argparse
from tqdm import tqdm
import scipy


def parse_args():
    parser = argparse.ArgumentParser(description='Convert Faster-RCNN results to json file for plotting')
    parser.add_argument(
        '--outdir', help="output dir for json results file", default=None, type=str)
    parser.add_argument(
        '--coco_anno_file', help="json file with coco style annotations for image set",
        default=None, type=str)
    parser.add_argument(
        '--pred_file', help="json file with Faster-RCNN predictions",
        default=None, type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def bbox_to_roi(bbox):
	roi_coords = [bbox[1], bbox[0], bbox[1]+bbox[3], bbox[0]+bbox[2]]
	return roi_coords

def parse_anno(anno_dict):
	img_dict = {}
	for img in anno_dict['images']:
		img_dict[img['id']] = img['file_name'][:11]
	return img_dict


def parse_pred(pred_list):
	roi_dict = {}
	for pred in pred_list:
		bbox = pred['bbox']
		roi_coords = bbox_to_roi(bbox)
		if pred['image_id'] not in roi_dict:
			roi_dict[pred['image_id']] = []
		value = {}
		value['roi'] = roi_coords
		value['score'] = pred['score']
		value['category_id'] = pred['category_id']
		roi_dict[pred['image_id']].append(value)
	return roi_dict


def convert_results(anno_file, pred_file, out_dir):
	json_data = open(anno_file).read()
	anno_dict = json.loads(json_data)
	json_data = open(pred_file).read()
	pred_list = json.loads(json_data)

	img_dict = parse_anno(anno_dict)
	roi_dict = parse_pred(pred_list)

	roi_dict_new = {}
	for img_id, img_name in img_dict.items():
		if img_id in roi_dict:
			roi_dict_new[img_name] = roi_dict[img_id]
		else:
			print("No detections in image {}".format(img_name))

	with open(os.path.join(out_dir, 'frcnn_detections_ycb.json'), 'wb') as outfile:
		outfile.write(json.dumps(roi_dict_new))


if __name__ == '__main__':
    args = parse_args()
    convert_results(args.coco_anno_file, args.pred_file, args.outdir)
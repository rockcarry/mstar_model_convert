# -*- coding: utf-8 -*-

import os
import json
import sys
import random
import shutil

def get_subset_gt(image_ids, gt_file, gt_subset_file='./subset_coco.json', box_only=False):
    gt0 = json.load(open(gt_file, 'r'))
    gt = dict()
    gt['licenses'] = gt0['licenses']
    gt['info'] = gt0['info']
    gt['categories'] = []
    gt['annotations'] = []
    gt['images'] = []
    gt_cat_ids = []
    for image in gt0['images']:
        if image['id'] in image_ids:
            gt['images'].append(image)
    for ann in gt0['annotations']:
        if ann['image_id'] in image_ids:
            gt['annotations'].append(ann)
            if ann['category_id'] not in gt_cat_ids:
                gt_cat_ids.append(ann['category_id'])
    for cat in gt0['categories']:
        if cat['id'] in gt_cat_ids:
            gt['categories'].append(cat)
    if box_only:
        with open(gt_subset_file, 'w') as fw:
            fw.write('[')
            for i, ann in enumerate(gt['annotations']):
                fw.write('{\"id\": ' + str(ann['id']))
                fw.write(', ' + '\"image_id\": ' + str(ann['image_id']))
                fw.write(', ' + '\"bbox\": [')
                for idx, b in enumerate(ann['bbox']):
                    if idx < len(ann['bbox']) - 1:
                        fw.write(str(b) + ',')
                    else:
                        fw.write(str(b) + ']')
                fw.write(', ' + '\"category_id\": ' + str(ann['category_id']))
                for cat in gt['categories']:
                    if cat['id'] == ann['category_id']:
                        fw.write(', \"category_name\": \"' + cat['name'] + '\"')
                if i < len(gt['annotations']) - 1:
                    fw.write('},\n')
                else:
                    fw.write('}]\n')
    else:
        with open(gt_subset_file, 'w') as fw:
            json.dump(gt, fw, indent=4, sort_keys=True)


def save_file(source_folder, dest_folder, num_files, gt_file):
    images = os.listdir(source_folder)
    images = [os.path.join(source_folder, i) for i in images]
    random.shuffle(images)
    copy_images = images[:num_files]
    if not os.path.exists(dest_folder):
        os.mkdir(dest_folder)
    dest_images_folder = os.path.join(dest_folder, 'subset_images')
    gt_subset_file = os.path.join(dest_folder, 'subset_coco.json')
    os.mkdir(dest_images_folder)
    for image in copy_images:
        shutil.copy(image, dest_images_folder)
    images_id = [int(os.path.split(i)[-1].split('.')[-2]) for i in copy_images]
    get_subset_gt(images_id, gt_file, gt_subset_file=gt_subset_file)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(sys.argv[0], '<source_folder> <source_json> <dest_images_folder> <num_files>')
        sys.exit(1)
    source_folder = sys.argv[1]
    source_json = sys.argv[2]
    dest_image_folder = sys.argv[3]
    num_files = int(sys.argv[4])
    save_file(source_folder, dest_image_folder, num_files, source_json)
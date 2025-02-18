import torch
import numpy as np
import cv2
import json
import sys, os
sys.path.append(os.getcwd())

from utils.xml_analysis import *
import utils.chefer_clip as clip
from utils.IN_dataset import *
from my_util import calculate_ETA

from datetime import datetime
import argparse

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(f"ViT-B/{args.clip_bb_num}", device=device, jit=False)

    ff = open("../datasets/IN/imagenet-classes.txt", 'r')
    in_class = []
    for l in ff.readlines():
        in_class.append(l[:-1].split(", ")[0])

    if args.dataset == "in":
        in_train_dl, in_val_dl, L, image_paths = get_IN_train_val(
            1000, args.batchsize, args.batchsize_test, args.num_workers)
    elif args.dataset == "in-c":
        in_val_dl, L, image_paths = get_INc_train_val(
            1000, args.batchsize_test, variant_name=args.corrupt_name,
            M=args.corrupt_magnitude, NW=args.num_workers)
    else:
        raise NotImplementedError
    if args.mode == 'flcp':
        if "mGPUs" in args.ckpt_path:
            from collections import OrderedDict

            new_state_dict = OrderedDict()
            loaded_dict = torch.load(args.ckpt_path, map_location="cpu")
            for k, v in loaded_dict.items():
                name = k[7:]
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        else:
            checkpoint = torch.load(args.ckpt_path)
            model.load_state_dict(checkpoint)
    with open(args.pred_json_path, 'r') as file:
        pred_list = json.load(file)

    bbox_dict = read_bboxs(csv_path="../datasets/IN/LOC_val_solution.csv")

    ES_list = []
    ES_list_R = []
    for iter, labeled_batch in enumerate(in_val_dl):
        if iter >= 5:
            break
        images, labels = labeled_batch
        images = images.to(device)
        labels = labels.to(device)
        # Calculate features
        indices = pred_list[iter * images.shape[0]:(iter+1) * images.shape[0]]
        useful_exp_idx = [
            elem1 == elem2 for elem1, elem2 in zip(indices, labels.cpu().tolist())
        ]
        cur_batch_size = len(useful_exp_idx)
        for si in range(cur_batch_size):
            img = images[si].unsqueeze(0)
            text = "a photo of a " + in_class[labels.cpu().tolist()[si]]
            text = clip.tokenize([text]).to(device)
            _, R_image = interpret(model=model, image=img, texts=text, device=device)
            vis, img_f, xai_map = show_image_relevance(R_image[0], img, orig_image=None)

            iid = image_paths[iter * args.batchsize_test + si].rsplit('/', 1)[-1].split('.')[0]
            img_ori_size = cv2.imread(image_paths[iter * args.batchsize_test + si])
            anno_mask = np.zeros((224, 224), dtype=np.uint8)
            for box_str in bbox_dict[iid].split('n')[1:]:
                cls_idx, x_min, y_min, x_max, y_max = box_str[:-1].split(' ')
                ratio = img_ori_size.shape[0] / 256 \
                    if img_ori_size.shape[0] < img_ori_size.shape[1] else img_ori_size.shape[1] / 256
                after_x, after_y = img_ori_size.shape[0] / ratio, img_ori_size.shape[1] / ratio
                ax, ay = (after_x - 224) / 2, (after_y - 224) / 2
                cv2.rectangle(img_f, (int(float(x_min) / ratio - ay), int(float(y_min) / ratio - ax)),
                              (int(float(x_max) / ratio - ay), int(float(y_max) / ratio - ax)), (0, 255, 0), 2)
                cv2.rectangle(anno_mask, (int(float(x_min) / ratio - ay), int(float(y_min) / ratio - ax)),
                              (int(float(x_max) / ratio - ay), int(float(y_max) / ratio - ax)), 1, -1)
            rma = eval_rma(pred=xai_map, anno=anno_mask, cls_thres=0.5)
            ES_list.append(rma)
            if useful_exp_idx[si]:
                ES_list_R.append(rma)
        end_t = datetime.now()
        if iter > 0 and iter % args.log_freq == 0:
            print("iter:", iter, "/", len(in_val_dl),
                  "ETA:", str(calculate_ETA(start_t, end_t, len(in_val_dl), iter, 0, 1)))
        start_t = datetime.now()

    with open(args.rationale_json_path, 'w') as file:
        json.dump(ES_list, file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--dataset', type=str, choices=['in', 'in-c'],
                        help="Dataset options; in: ImageNet-1K val.; in-c: ImageNet-C.")
    parser.add_argument('--corrupt_name', type=str, default='',
                        help="ImageNet-C only, testing data corruption type.")
    parser.add_argument('--corrupt_magnitude', type=int, default=1,
                        help="ImageNet-C only, testing data corruption magnitude, choose from 1 to 5.")
    parser.add_argument('--clip_bb_num', type=int, choices=[16, 32],
                        help="Choose which type of CLIP model to use.")
    parser.add_argument('--batchsize', type=int, help="Batch size for fine-tuning.")
    parser.add_argument('--batchsize_test', type=int, help="Batch size for model evaluation.")
    parser.add_argument('--mode', type=str, choices=['zs', 'flcp'],
                        help="Mode for fine-tuning.")
    parser.add_argument('--ckpt_path', type=str, help="Path for checkpoint (FLCP mode).")
    parser.add_argument('--pred_json_path', type=str,
                        help="Path for json file which saves the prediction results.")
    parser.add_argument('--rationale_json_path', type=str,
                        help="Path for json file which saves the rationale results.")
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--log_freq', type=int, default=25)
    args = parser.parse_args()

    main(args)
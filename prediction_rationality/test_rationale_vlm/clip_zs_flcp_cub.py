import torch
import numpy as np
import cv2
import sys, os
sys.path.append(os.getcwd())

import utils.chefer_clip as clip
from utils.CUB_dataset import *
from utils.xml_analysis import *
from my_util import calculate_ETA

from datetime import datetime
import json
import argparse

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(f'ViT-B/{args.clip_bb_num}', device=device, jit=False)

    ff = open("../datasets/CUB/CUB_200_2011/classes.txt", 'r')
    cub_class = []
    for l in ff.readlines():
        cub_class.append(l[:-1].split(" ")[1].split(".")[1].replace("_", " ").lower())
    cub_test_dl, cub_seg_test_dl, L = get_CUB_train_test(
        args.batchsize, args.batchsize_test, args.num_workers, SEG=True)

    if args.mode == "flcp":
        checkpoint = torch.load(args.ckpt_path)
        model.load_state_dict(checkpoint)
    with open(args.pred_json_path, 'r') as file:
        pred_list = json.load(file)

    iter = 0
    ES_list = []
    ES_list_r = []
    for labeled_batch, labeled_seg_batch in zip(cub_test_dl, cub_seg_test_dl):
        images, labels = labeled_batch
        segments, _ = labeled_seg_batch
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
            text = "a photo of a " + cub_class[labels.cpu().tolist()[si]]
            text = clip.tokenize([text]).to(device)
            _, R_image = interpret(model=model, image=img, texts=text, device=device)
            vis, img_f, xai_map = show_image_relevance(R_image[0], img, orig_image=None)
            seg_mask = np.mean(segments[si].permute(1, 2, 0).numpy(), axis=2).astype(np.uint8)
            rma = eval_rma(pred=xai_map, anno=seg_mask, cls_thres=0.5)
            ES_list.append(rma)
            if useful_exp_idx[si]:
                ES_list_r.append(rma)
        end_t = datetime.now()
        if iter > 0:
            print("iter:", iter, "/", len(cub_test_dl),
                  "ETA:", str(calculate_ETA(start_t, end_t, len(cub_test_dl), iter, 0, 1)))
        start_t = datetime.now()
        iter += 1

    with open(args.rationale_json_path, 'w') as file:
        json.dump(ES_list, file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
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
    args = parser.parse_args()

    main(args)
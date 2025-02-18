import torch
import numpy as np
from datetime import datetime
import cv2
import json
import sys, os
sys.path.append(os.getcwd())

from utils.xml_analysis_lp import *
from utils.lp_model import LP_Net
import utils.chefer_clip as clip
from utils.caltech101_dataset import *
from my_util import calculate_ETA
import argparse

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load(f"ViT-B/{args.clip_bb_num}", device=device, jit=False)
    model = LP_Net(clip_model, 512, 102, freeze_clip=False).to(device)
    checkpoint = torch.load(args.ckpt_path)
    model.load_state_dict(checkpoint)

    with open('../datasets/caltech_101/label_dict.json', 'r') as file:
        label_dict = json.load(file)
    ct_class = []
    for k, v in label_dict.items():
        ct_class.append(v)
    ct_train_dl, ct_test_dl, L = get_CT101_train_val(
        args.batchsize, args.batchsize_test, args.num_workers, test_ratio=0.2)

    iter = 0
    ES_list = []
    ES_list_R = []
    for images, labels, _, bbox_masks in ct_test_dl:
        segments = bbox_masks
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.shape[0]
        for i in range(batch_size):
            seg_mask = segments[i].numpy().astype(np.uint8)
            vis_f, rma, pred_idx = generate_visualization(
                model, ct_class, seg_mask, images[i], -1.0, 0.5, class_index=labels[i].item(), energy=True
            )
            ES_list.append(rma)
            if pred_idx == labels[i].item():
                ES_list_R.append(rma)
        end_t = datetime.now()
        if iter > 0:
            print("iter:", iter, "/", len(ct_test_dl),
                  "ETA:", str(calculate_ETA(start_t, end_t, len(ct_test_dl), iter, 0, 1)))
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
    parser.add_argument('--mode', type=str, choices=['lp', 'ft'], help="Mode for fine-tuning.")
    parser.add_argument('--ckpt_path', type=str, help="Path for checkpoint.")
    parser.add_argument('--pred_json_path', type=str,
                        help="Path for json file which saves the prediction results.")
    parser.add_argument('--rationale_json_path', type=str,
                        help="Path for json file which saves the rationale results.")
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()

    main(args)
import torch
import numpy as np
import cv2

import sys, os
sys.path.append(os.getcwd())

from utils.IN_dataset import *
from ALBEF.ALBEF import ALBEF
from ALBEF.tokenization_bert import BertTokenizer
from ALBEF.vit import interpolate_pos_embed
from utils.xml_analysis import *

import argparse
try:
    import ruamel_yaml as yaml
except:
    import ruamel.yaml as yaml
from pathlib import Path
from datetime import datetime
import json

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='../rethinking_ft_vlm/ALBEF/configs/ours.yaml')
parser.add_argument('--checkpoint', default='../my_ckpts/ALBEF.pth')
parser.add_argument('--resume', default=False, type=bool)
parser.add_argument('--output_dir', default='./ALBEF_output/')
parser.add_argument('--text_encoder', default='bert-base-uncased')
parser.add_argument('--device', default='cuda')
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
parser.add_argument('--distributed', default=False, type=bool)
args = parser.parse_args()

config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
Path(args.output_dir).mkdir(parents=True, exist_ok=True)
yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

"""START of Custom configurations"""
BATCH_SIZE = 128  # batch size for data loading.
NUM_WORKERS = 8  # num. of workers for dataloader.
LOG_FREQ = 50  # iteration frequency of model logging.
MGPU = True  # indicating whether your loaded weights from FLCP were trained with multiple GPUs or not.
FLCP = True  # indicating evaluation with whether "ZS" (False) or "FLCP" (True) method.
FLCP_CKPT_PATH = ""  # REQUIRED: if evaluation with FLCP, input your path for checkpoint weights.
PRED_JSON_PATH = ""  # REQUIRED: path for loading json file which includes model prediction indices.
BINARY_RMA = True  # if False, you will get the continous RMA score, if True, you will get binary 0-1 RMA value for each sample (L142 of our paper).
CLS_THRES = 0.5 if BINARY_RMA else -1.0  # natural threshold for getting binary 0-1 RMA values.
BINARY_RMA_JSON_PATH = ""  # REQUIRED: path for storing json file which includes binary 0-1 RMA values.
#
# Note1: if you want to do PECM, PT and IR evaluations later, "BINARY_RMA_JSON_PATH" is required.
# Note2: for "PRED_JSON_PATH", if FLCP = True, loading prediction indices of FLCP, otherwise loading prediction indices of ZS.
#
"""END of Custom configurations"""

#### Model ####
print("Creating model")
model = ALBEF(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
if args.checkpoint:
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state_dict = checkpoint['model']
    pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], model.visual_encoder)
    state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
    if config['alpha'] >= 0:
        m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'], model.visual_encoder_m)
        state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped
    for key in list(state_dict.keys()):
        if 'bert' in key:
            encoder_key = key.replace('bert.', '')
            state_dict[encoder_key] = state_dict[key]
            del state_dict[key]
    msg = model.load_state_dict(state_dict, strict=False)

if FLCP:
    ckpt_path = FLCP_CKPT_PATH
    if MGPU:
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        loaded_dict = torch.load(ckpt_path, map_location="cpu")
        for k, v in loaded_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint)
with open(PRED_JSON_PATH, 'r') as file:
    pred_list = json.load(file)

ff = open("../imagenet-classes.txt", 'r')
in_class = []
for l in ff.readlines():
    in_class.append(l[:-1].split(", ")[0])
bbox_dict = read_bboxs(csv_path="../datasets/kaggle_IN1K_loc/LOC_val_solution.csv")
in_train_dl, in_val_dl, L, image_paths = get_IN_train_val(1000, BATCH_SIZE, BATCH_SIZE, NUM_WORKERS)
text_inputs = tokenizer(
    [f"a photo of a {c}" for c in in_class], padding='max_length', truncation=True, max_length=30, return_tensors="pt"
).to(device)

iter = 0
ES_list = []
ES_list_r = []
for iter, labeled_batch in enumerate(in_val_dl):
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
        text = tokenizer([text], padding='max_length', truncation=True, max_length=30, return_tensors="pt").to(device)
        _, R_image = interpret_albef_blip(model=model, image=img, texts=text, device=device)
        vis, img_f, xai_map = show_image_relevance(R_image[0], img, orig_image=None)

        iid = image_paths[iter * BATCH_SIZE + si].rsplit('/', 1)[-1].split('.')[0]
        img_ori_size = cv2.imread(image_paths[iter * BATCH_SIZE + si])
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

        rma = eval_rma(pred=xai_map, anno=anno_mask, cls_thres=CLS_THRES)
        ES_list.append(rma)
        if useful_exp_idx[si]:
            ES_list_r.append(rma)

    end_t = datetime.now()
    if iter > 0 and iter % LOG_FREQ == 0:
        print("iter:", iter, "/", len(in_val_dl),
              "ETA:", str(calculate_ETA(start_t, end_t, len(in_val_dl), iter, 1)))
    start_t = datetime.now()

if BINARY_RMA:
    with open(BINARY_RMA_JSON_PATH, 'w') as file:
        json.dump(ES_list, file)
else:
    print("RMA Score for Right Samples:", sum(ES_list_r) / len(ES_list_r))
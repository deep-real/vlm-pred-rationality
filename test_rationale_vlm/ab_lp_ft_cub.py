import torch
import numpy as np
import sys, os
sys.path.append(os.getcwd())

from utils.xml_analysis_lp import *
from utils.lp_model import LP_Net_A
from utils.CUB_dataset import *

import argparse
try:
    import ruamel_yaml as yaml
except:
    import ruamel.yaml as yaml
from pathlib import Path
from datetime import datetime
import os
import cv2
import json

from ALBEF.ALBEF import ALBEF
from ALBEF.tokenization_bert import BertTokenizer
from ALBEF.vit import interpolate_pos_embed

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
LOG_FREQ = 10  # iteration frequency of model logging.
FT = True  # indicating evaluation with whether "LP" (False) or "FT" (True) method.
CKPT_PATH = ""  # REQUIRED: input your path for checkpoint weights.
BINARY_RMA = True  # if False, you will get the continous RMA score, if True, you will get binary 0-1 RMA value for each sample (L142 of our paper).
CLS_THRES = 0.5 if BINARY_RMA else -1.0  # natural threshold for getting binary 0-1 RMA values.
BINARY_RMA_JSON_PATH = ""  # REQUIRED: path for storing json file which includes binary 0-1 RMA values.
#
# Note: if you want to do PECM, PT and IR evaluations later, "BINARY_RMA_JSON_PATH" is required.
#
"""END of Custom configurations"""

#### Model ####
print("Creating model")
albef_model = ALBEF(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer)
device = "cuda" if torch.cuda.is_available() else "cpu"
albef_model = albef_model.to(device)
if args.checkpoint:
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state_dict = checkpoint['model']
    pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], albef_model.visual_encoder)
    state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
    if config['alpha'] >= 0:
        m_pos_embed_reshaped = interpolate_pos_embed(
            state_dict['visual_encoder_m.pos_embed'], albef_model.visual_encoder_m
        )
        state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped
    for key in list(state_dict.keys()):
        if 'bert' in key:
            encoder_key = key.replace('bert.', '')
            state_dict[encoder_key] = state_dict[key]
            del state_dict[key]
    msg = albef_model.load_state_dict(state_dict, strict=False)

model = LP_Net_A(albef_model, 768, 200, freeze_ab=False).to(device)
# LP_Net_A is for both "LP" and "FT" methods since they have identical model structure.
model_path = CKPT_PATH
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint)

ff = open("../datasets/CUB/CUB_200_2011/classes.txt", 'r')
cub_class = []
for l in ff.readlines():
    cub_class.append(l[:-1].split(" ")[1].split(".")[1].replace("_", " ").lower())
cub_test_dl, cub_seg_test_dl, L = get_CUB_train_test(BATCH_SIZE, BATCH_SIZE, NUM_WORKERS, SEG=True)
text_inputs = tokenizer(
    [f"a photo of a {c}" for c in cub_class], padding='max_length', truncation=True, max_length=30, return_tensors="pt"
).to(device)

iter = 0
ES_list = []
ES_list_R = []
for labeled_batch, labeled_seg_batch in zip(cub_test_dl, cub_seg_test_dl):
    images, labels = labeled_batch
    segments, _ = labeled_seg_batch
    images = images.to(device)
    labels = labels.to(device)
    batch_size = labels.shape[0]
    for i in range(batch_size):
        seg_mask = np.mean(segments[i].permute(1, 2, 0).numpy(), axis=2).astype(np.uint8)
        vis_f, rma, pred_idx = generate_visualization_ab(
            model, cub_class, seg_mask, images[i], -1.0, CLS_THRES, class_index=labels[i].item(), energy=True
        )
        ES_list.append(rma)
        if pred_idx == labels[i].item():
            ES_list_R.append(rma)
    end_t = datetime.now()
    if iter > 0 and iter % LOG_FREQ == 0:
        print("iter:", iter, "/", len(cub_test_dl),
              "ETA:", str(calculate_ETA(start_t, end_t, len(cub_test_dl), iter, 1)))
    start_t = datetime.now()
    iter += 1

if BINARY_RMA:
    with open(BINARY_RMA_JSON_PATH, 'w') as file:
        json.dump(ES_list, file)
else:
    print("RMA Score for Right Samples:", sum(ES_list_R) / len(ES_list_R))
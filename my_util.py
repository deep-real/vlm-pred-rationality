import torch
from torch import optim
from datetime import datetime
try:
    import ruamel_yaml as yaml
except:
    import ruamel.yaml as yaml

import sys, os
# sys.path.append(os.path.join(os.getcwd(), 'utils'))
# print(sys.path)
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from ALBEF.ALBEF import ALBEF
from ALBEF.tokenization_bert import BertTokenizer
from ALBEF.vit import interpolate_pos_embed
from BLIP.blip_cls import blip_cls
import clip

from utils.CUB_dataset import *
import utils
# from utils import CUB_dataset
from utils.Stanford_Dogs_dataset import *
from utils.caltech101_dataset import *
from utils.IN_dataset import *
# CUB_dataset.get_CUB_train_test(64, 64, 8)

def load_albef_model(args):
    config = yaml.load(open(args.config_albef, 'r'), Loader=yaml.Loader)
    tokenizer = BertTokenizer.from_pretrained(args.albef_text_encoder)
    model = ALBEF(config=config, text_encoder=args.albef_text_encoder, tokenizer=tokenizer)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    if args.albef_ori_ckpt:
        checkpoint = torch.load(args.albef_ori_ckpt, map_location='cpu')
        state_dict = checkpoint['model']
        pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], model.visual_encoder)
        state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
        if config['alpha'] >= 0:
            m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],
                                                         model.visual_encoder_m)
            state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped
        for key in list(state_dict.keys()):
            if 'bert' in key:
                encoder_key = key.replace('bert.', '')
                state_dict[encoder_key] = state_dict[key]
                del state_dict[key]
        msg = model.load_state_dict(state_dict, strict=False)
    if args.epoch > 0:
        current_time = datetime.now()
        time_string = current_time.strftime("%Y-%m-%d_%H:%M:%S")
        fn = (time_string + f"_albef(ViT-B-16)_{args.mode}_{args.dataset}_e" + str(args.epoch) +
              "_bs" + str(args.batchsize) + "_lr" + str(args.learning_rate))
        if not os.path.exists(os.path.join("../my_ckpts", fn)):
            os.mkdir(os.path.join("../my_ckpts", fn))
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate,
                               betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    else:
        optimizer, fn = None, None
    return model, optimizer, fn, tokenizer

def load_blip_model(args):
    config = yaml.load(open(args.config_blip, 'r'), Loader=yaml.Loader)
    model = blip_cls(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'],
                     vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'],
                     negative_all_rank=config['negative_all_rank'], med_config=config['med_config'])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    if args.epoch > 0:
        current_time = datetime.now()
        time_string = current_time.strftime("%Y-%m-%d_%H:%M:%S")
        fn = (time_string + f"_blip(ViT-B-16)_{args.mode}_{args.dataset}_e" + str(args.epoch)
              + "_bs" + str(args.batchsize) + "_lr" + str(args.learning_rate))
        if not os.path.exists(os.path.join("../my_ckpts", fn)):
            os.mkdir(os.path.join("../my_ckpts", fn))
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8,
                               weight_decay=0.0)
    else:
        optimizer, fn = None, None
    return model, optimizer, fn

def load_clip_model(epoch, bs, lr, bb_num, dataset_name, mode):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load(f"ViT-B/{bb_num}", device=device, jit=False)
    if epoch > 0:
        current_time = datetime.now()
        time_string = current_time.strftime("%Y-%m-%d_%H:%M:%S")
        fn = (time_string + f"_clip(ViT-B-{bb_num})_{mode}_{dataset_name}_e" +
              str(epoch) + "_bs" + str(bs) + "_lr" + str(lr))
        if not os.path.exists(os.path.join("../my_ckpts", fn)):
            os.mkdir(os.path.join("../my_ckpts", fn))
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    else:
        optimizer, fn = None, None
    return model, optimizer, fn

def load_cub_dataset(bs, test_bs, n_w):
    ff = open("../datasets/CUB/CUB_200_2011/classes.txt", 'r')
    cub_class = []
    for l in ff.readlines():
        cub_class.append(l[:-1].split(" ")[1].split(".")[1].replace("_", " ").lower())
    cub_train_dl, cub_test_dl, L = get_CUB_train_test(bs, test_bs, n_w)
    return cub_train_dl, cub_test_dl, L, cub_class

def load_sd_dataset(bs, test_bs, n_w):
    with open('../datasets/Stanford_Dogs/label_dict.json', 'r') as file:
        label_dict = json.load(file)
    sd_class = []
    for k, v in label_dict.items():
        sd_class.append(v)
    sd_train_dl, sd_test_dl, L = get_StanDog_train_val(bs, test_bs, n_w)
    return sd_train_dl, sd_test_dl, L, sd_class

def load_ct_dataset(bs, test_bs, n_w):
    with open('../datasets/caltech_101/label_dict.json', 'r') as file:
        label_dict = json.load(file)
    ct_class = []
    for k, v in label_dict.items():
        ct_class.append(v)
    ct_train_dl, ct_test_dl, L = get_CT101_train_val(bs, test_bs, n_w, test_ratio=0.2)
    return ct_train_dl, ct_test_dl, L, ct_class

def load_in_dataset(bs, test_bs, n_w):
    ff = open("../datasets/IN/imagenet-classes.txt", 'r')
    in_class = []
    for l in ff.readlines():
        # in_class.append(l[:-1])
        in_class.append(l[:-1].split(", ")[0])
    in_train_dl, in_val_dl, L, image_paths = get_IN_train_val(1000, bs, test_bs, n_w)
    return in_train_dl, in_val_dl, L, in_class, image_paths

def load_inc_dataset(test_bs, vn, vn_m, n_w):
    ff = open("../datasets/IN/imagenet-classes.txt", 'r')
    in_class = []
    for l in ff.readlines():
        in_class.append(l[:-1].split(", ")[0])
    in_val_dl, L, image_paths = get_INc_train_val(1000, test_bs, variant_name=vn, M=vn_m, NW=n_w)
    return in_val_dl, L, in_class, image_paths

def calculate_ETA(start_t, end_t, len_loader, iter, cur_epoch, epoch):
    current_t = (end_t - start_t) * (len_loader - iter)
    next_t = (end_t - start_t) * len_loader * (epoch - cur_epoch - 1)
    return current_t + next_t

def label2text2embedding(labels, class_list, device):
    text_inputs = torch.cat(
        [clip.tokenize(f"a photo of a {class_list[l]}") for l in labels.cpu().tolist()]
    ).to(device)
    return text_inputs

def label2text2embedding_albef(labels, class_list, device, tokenizer):
    inputs = [f"a photo of a {class_list[l]}" for l in labels.cpu().tolist()]
    text_inputs = tokenizer(
        inputs, padding='max_length', truncation=True, max_length=30, return_tensors="pt"
    ).to(device)
    return text_inputs

def label2text_blip(labels, class_list):
    inputs = [f"a photo of a {class_list[l]}" for l in labels.cpu().tolist()]
    return inputs

def text2embedding(texts, device):
    text_inputs = torch.cat([clip.tokenize(f"{t}") for t in texts]).to(device)
    return text_inputs

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        try:
            p.grad.data = p.grad.data.float()
        except:
            pass

def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
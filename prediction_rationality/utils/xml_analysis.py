import torch
import utils.chefer_clip as clip
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
# from captum.attr import visualization
import csv

def interpret(image, texts, model, device, start_layer=-1, start_layer_text=-1):
    batch_size = texts.shape[0]
    images = image.repeat(batch_size, 1, 1, 1)

    logits_per_image, logits_per_text = model(images, texts)
    probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
    index = [i for i in range(batch_size)]
    one_hot = np.zeros((logits_per_image.shape[0], logits_per_image.shape[1]), dtype=np.float32)
    one_hot[torch.arange(logits_per_image.shape[0]), index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.cuda() * logits_per_image)
    model.zero_grad()

    image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())

    if start_layer == -1:
        start_layer = len(image_attn_blocks) - 1

    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
    R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
    R = R.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
    for i, blk in enumerate(image_attn_blocks):
        if i < start_layer:
            continue
        grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
        cam = blk.attn_probs.detach()
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0).mean(dim=1)
        R = R + torch.bmm(cam, R)
    image_relevance = R[:, 0, 1:]

    return None, image_relevance

def interpret_albef_blip(image, texts, model, device, start_layer=-1, start_layer_text=-1):
    batch_size = 1
    images = image.repeat(batch_size, 1, 1, 1)
    image_feats = model.visual_encoder(images, exp_mode=True)
    image_embeds = model.vision_proj(image_feats[:, 0, :])
    image_embeds = torch.nn.functional.normalize(image_embeds, dim=-1)
    text_outputs = model.text_encoder(texts.input_ids, attention_mask=texts.attention_mask, mode='text')
    text_feats = text_outputs.last_hidden_state
    text_embeds = torch.nn.functional.normalize(model.text_proj(text_feats[:, 0, :]))
    logits_per_image = image_embeds @ text_embeds.t() / model.temp
    index = [i for i in range(batch_size)]
    one_hot = np.zeros((logits_per_image.shape[0], logits_per_image.shape[1]), dtype=np.float32)
    one_hot[torch.arange(logits_per_image.shape[0]), index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.cuda() * logits_per_image)
    model.zero_grad()

    image_attn_blocks = list(dict(model.visual_encoder.blocks.named_children()).values())

    if start_layer == -1:
        start_layer = len(image_attn_blocks) - 1

    num_tokens = image_attn_blocks[0].attn.attention_map.shape[-1]
    R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn.attention_map.dtype).to(device)
    R = R.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
    for i, blk in enumerate(image_attn_blocks):
        if i < start_layer:
            continue
        grad = torch.autograd.grad(one_hot, [blk.attn.attention_map], retain_graph=True)[0].detach()
        cam = blk.attn.attention_map.detach()
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0).mean(dim=1)
        R = R + torch.bmm(cam, R)
    image_relevance = R[:, 0, 1:]

    return None, image_relevance

def show_cam_on_image(img, mask, colormap_type=cv2.COLORMAP_JET):
    if colormap_type == cv2.COLORMAP_JET:
        heatmap = cv2.applyColorMap(np.uint8(255 * (1 - mask)), colormap_type)
    else:
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap_type)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam

def show_image_relevance(image_relevance, image, orig_image):
    dim = int(image_relevance.numel() ** 0.5)
    image_relevance = image_relevance.reshape(1, 1, dim, dim)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bilinear')
    image_relevance = image_relevance.reshape(224, 224).cuda().data.cpu().numpy()
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    image = image[0].permute(1, 2, 0).data.cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())
    vis = show_cam_on_image(image, image_relevance)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis, cv2.cvtColor(np.uint8(255 * image), cv2.COLOR_RGB2BGR), image_relevance

def show_mask_relevance(xai_mask, image):
    image = image[0].permute(1, 2, 0).data.cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())
    vis = show_cam_on_image(image, xai_mask, colormap_type=cv2.COLORMAP_OCEAN)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis

from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()

def calculate_ETA(start_t, end_t, len_loader, iter, epoch):
    current_t = (end_t - start_t) * (len_loader - iter)
    next_t = (end_t - start_t) * len_loader
    return current_t

def read_bboxs(csv_path):
    R = 0
    res_dict = {}
    with open(csv_path, encoding='utf-8-sig') as f:
        for row in csv.reader(f, skipinitialspace=True):
            if R == 0:
                R += 1
                continue
            R += 1
            res_dict[row[0]] = row[1]
    f.close()
    return res_dict

def eval_rma(pred, anno, cls_thres=-1.0):
    es = np.sum(pred * anno) / np.sum(pred)
    if cls_thres < 0:
        return es
    else:
        return 0 if es < cls_thres else 1

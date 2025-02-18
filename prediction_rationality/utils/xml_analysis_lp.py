import torch
import numpy as np
import cv2

def avg_heads(cam, grad):
    cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
    grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
    cam = grad * cam
    cam = cam.clamp(min=0).mean(dim=0)
    return cam

def apply_self_attention_rules(R_ss, cam_ss):
    R_ss_addition = torch.matmul(cam_ss, R_ss)
    return R_ss_addition

def generate_relevance(model, input, class_index):
    output = model(input)
    pred_index = np.argmax(output.cpu().data.numpy(), axis=-1)

    one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
    one_hot[0, class_index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.cuda() * output)
    model.zero_grad()

    image_attn_blocks = list(
        dict(model.clip_feature_extractor.visual.transformer.resblocks.named_children()).values()
    )

    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
    R = torch.eye(num_tokens, num_tokens).cuda()
    for blk in image_attn_blocks:
        grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
        cam = blk.attn_probs.detach()
        cam = avg_heads(cam, grad)
        R += apply_self_attention_rules(R.cuda(), cam.cuda())
    return R[0, 1:], pred_index[0]

def generate_relevance_ab(model, input, class_index):
    output = model(input, exp_mode=True)
    pred_index = np.argmax(output.cpu().data.numpy(), axis=-1)

    one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
    one_hot[0, class_index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.cuda() * output)
    model.zero_grad()
    image_attn_blocks = list(dict(model.ab_feature_extractor.blocks.named_children()).values())

    num_tokens = image_attn_blocks[0].attn.get_attention_map().shape[-1]
    R = torch.eye(num_tokens, num_tokens).cuda()
    for blk in image_attn_blocks:
        grad = torch.autograd.grad(one_hot, [blk.attn.attention_map], retain_graph=True)[0].detach()
        cam = blk.attn.attention_map.detach()
        cam = avg_heads(cam, grad)
        R += apply_self_attention_rules(R.cuda(), cam.cuda())
    return R[0, 1:], pred_index[0]

def generate_relevance_2(model, input, anno_index):
    output = model(input)
    pred_index = np.argmax(output.cpu().data.numpy(), axis=-1)

    one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
    one_hot[0, anno_index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.cuda() * output)

    model.zero_grad()
    one_hot.backward(retain_graph=True)

    image_attn_blocks = list(
        dict(model.clip_feature_extractor.visual.transformer.resblocks.named_children()).values()
    )

    # num_tokens = image_attn_blocks[0].attn.get_attention_map().shape[-1]
    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
    R = torch.eye(num_tokens, num_tokens).cuda()
    for blk in image_attn_blocks:
        grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
        cam = blk.attn_probs.detach()
        cam = avg_heads(cam, grad)
        R += apply_self_attention_rules(R.cuda(), cam.cuda())
    return R[0, 1:], pred_index[0]

# create heatmap from mask on image
def show_cam_on_image(img, mask, colormap_type=cv2.COLORMAP_JET):
    if colormap_type == cv2.COLORMAP_JET:
        heatmap = cv2.applyColorMap(np.uint8(255 * (1 - mask)), colormap_type)
    else:
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap_type)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam

def show_mask_relevance(xai_mask, image):
    vis = show_cam_on_image(image, xai_mask, colormap_type=cv2.COLORMAP_OCEAN)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis

def generate_visualization(model, class_list, anno_mask, original_image,
                           heatmap_thres, cls_thres, class_index=None, energy=True):
    transformer_attribution, pred_idx = generate_relevance(
        model, original_image.unsqueeze(0).cuda().detach(), class_index)
    dim = int(transformer_attribution.numel() ** 0.5)
    transformer_attribution = transformer_attribution.reshape(1, 1, dim, dim)
    transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, size=224, mode='bilinear')
    transformer_attribution = transformer_attribution.reshape(224, 224).cuda().data.cpu().numpy()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / \
                              (transformer_attribution.max() - transformer_attribution.min())
    image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()
    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (
                image_transformer_attribution.max() - image_transformer_attribution.min())
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    img_f = np.uint8(255 * image_transformer_attribution)
    img_f = cv2.cvtColor(np.array(img_f), cv2.COLOR_RGB2BGR)

    anno_mask_f = cv2.cvtColor(anno_mask * 255, cv2.COLOR_GRAY2BGR)

    vis_f = cv2.vconcat([img_f, vis, anno_mask_f])

    if energy:
        from utils.xml_analysis import eval_rma
        score = eval_rma(pred=transformer_attribution, anno=anno_mask, cls_thres=cls_thres)
    else:
        raise NotImplementedError
    return vis_f, score, pred_idx

def generate_visualization_ab(model, class_list, anno_mask, original_image,
                              heatmap_thres, cls_thres, class_index=None, energy=True):
    transformer_attribution, pred_idx = generate_relevance_ab(
        model, original_image.unsqueeze(0).cuda().detach(), class_index)
    dim = int(transformer_attribution.numel() ** 0.5)
    transformer_attribution = transformer_attribution.reshape(1, 1, dim, dim)
    transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, size=224, mode='bilinear')
    transformer_attribution = transformer_attribution.reshape(224, 224).cuda().data.cpu().numpy()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / \
                              (transformer_attribution.max() - transformer_attribution.min())
    image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()
    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (
                image_transformer_attribution.max() - image_transformer_attribution.min())
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    img_f = np.uint8(255 * image_transformer_attribution)
    img_f = cv2.cvtColor(np.array(img_f), cv2.COLOR_RGB2BGR)
    anno_mask_f = cv2.cvtColor(anno_mask * 255, cv2.COLOR_GRAY2BGR)

    vis_f = cv2.vconcat([img_f, vis, anno_mask_f])
    if energy:
        from utils.xml_analysis import eval_rma
        score = eval_rma(pred=transformer_attribution, anno=anno_mask, cls_thres=cls_thres)
    else:
        raise NotImplementedError
    return vis_f, score, pred_idx

def generate_visualization_IN(model, original_image, class_index, energy=True):
    transformer_attribution, pred_idx = generate_relevance(
        model, original_image.unsqueeze(0).cuda().detach(), class_index)
    dim = int(transformer_attribution.numel() ** 0.5)
    transformer_attribution = transformer_attribution.reshape(1, 1, dim, dim)
    transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, size=224, mode='bilinear')
    transformer_attribution = transformer_attribution.reshape(224, 224).cuda().data.cpu().numpy()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / \
                              (transformer_attribution.max() - transformer_attribution.min())
    image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()
    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (
                image_transformer_attribution.max() - image_transformer_attribution.min())
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    img_f = np.uint8(255 * image_transformer_attribution)
    img_f = cv2.cvtColor(np.array(img_f), cv2.COLOR_RGB2BGR)

    return vis, img_f, transformer_attribution, pred_idx

def generate_visualization_IN_ab(model, original_image, class_index, energy=True):
    transformer_attribution, pred_idx = generate_relevance_ab(
        model, original_image.unsqueeze(0).cuda().detach(), class_index)
    dim = int(transformer_attribution.numel() ** 0.5)
    transformer_attribution = transformer_attribution.reshape(1, 1, dim, dim)
    transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, size=224, mode='bilinear')
    transformer_attribution = transformer_attribution.reshape(224, 224).cuda().data.cpu().numpy()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / \
                              (transformer_attribution.max() - transformer_attribution.min())
    image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()
    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (
                image_transformer_attribution.max() - image_transformer_attribution.min())
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    img_f = np.uint8(255 * image_transformer_attribution)
    img_f = cv2.cvtColor(np.array(img_f), cv2.COLOR_RGB2BGR)

    return vis, img_f, transformer_attribution, pred_idx

def calculate_ETA(start_t, end_t, len_loader, iter, epoch):
    current_t = (end_t - start_t) * (len_loader - iter)
    next_t = (end_t - start_t) * len_loader
    return current_t
import torch
import sys, os
sys.path.append(os.getcwd())
from utils.lp_model import *

from my_util import *
import json
import argparse

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # dataset
    if args.dataset == "cub":
        data_train_dl, data_test_dl, L, class_list = load_cub_dataset(
            args.batchsize, args.batchsize_test, args.num_workers
        )
        cls_num = 200
    elif args.dataset == "sd":
        data_train_dl, data_test_dl, L, class_list = load_sd_dataset(
            args.batchsize, args.batchsize_test, args.num_workers
        )
        cls_num = 120
    elif args.dataset == "ct":
        data_train_dl, data_test_dl, L, class_list = load_ct_dataset(
            args.batchsize, args.batchsize_test, args.num_workers
        )
        cls_num = 102
    elif args.dataset == "in":
        data_train_dl, data_test_dl, L, class_list, image_paths = load_in_dataset(
            args.batchsize, args.batchsize_test, args.num_workers
        )
        cls_num = 1000
    elif args.dataset == "in-c":
        data_test_dl, L, class_list, image_paths = load_inc_dataset(
            args.batchsize_test, args.corrupt_name, args.corrupt_magnitude, args.num_workers
        )
        cls_num = 1000
    else:
        raise NotImplementedError
    # model
    ft_whole_model = True if args.mode == 'ft' else False
    if args.vlm == "albef16":
        vlm_model, optimizer, fn, tokenizer = load_albef_model(args)
        model = LP_Net_A(vlm_model, 768, cls_num, freeze_ab=not ft_whole_model).to(device)
    elif args.vlm == "blip16":
        vlm_model, optimizer, fn = load_blip_model(args)
        model = LP_Net_A(vlm_model, 768, cls_num, freeze_ab=not ft_whole_model).to(device)
    elif "clip" in args.vlm:
        vlm_model, optimizer, fn = load_clip_model(
            args.epoch, args.batchsize, -1.0, bb_num=int(args.vlm[-2:]),
            dataset_name=args.dataset, mode=args.mode
        )
        model = LP_Net(vlm_model, 512, cls_num, freeze_clip=not ft_whole_model).to(device)
    else:
        raise NotImplementedError

    if "mGPUs" in args.ckpt_path:
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        loaded_dict = torch.load(args.ckpt_path, map_location="cpu")
        for k, v in loaded_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        print("Loaded my mGPUs ft model.")
    else:
        checkpoint = torch.load(args.ckpt_path)
        model.load_state_dict(checkpoint)

    print("Validation")
    model.eval()
    R = 0
    pred_list_all = []
    for iter, labeled_batch in enumerate(data_test_dl):
        if args.dataset in ["cub", "in", "in-c"]:
            images, labels = labeled_batch
        else:
            images, labels, _, _ = labeled_batch
        images = images.to(device)
        labels = labels.to(device)
        logits_pred = model(images)
        pred_list = torch.argmax(logits_pred, dim=1).cpu().tolist()
        anno_list = labels.cpu().tolist()
        pred_list_all += pred_list
        R += sum(1 for x, y in zip(pred_list, anno_list) if x == y)
    print("acc:", float(R) / L)

    with open(args.json_path, 'w') as file:
        json.dump(pred_list_all, file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--dataset', type=str, choices=['cub', 'sd', 'ct', 'in', 'in-c'],
                        help="Dataset options; cub: CUB-200-2011, sd: Stanford-Dogs, "
                             "caltech: CalTech101; in: ImageNet-1K val.; in-c: ImageNet-C.")
    parser.add_argument('--corrupt_name', type=str, default='',
                        help="ImageNet-C only, testing data corruption type.")
    parser.add_argument('--corrupt_magnitude', type=int, default=1,
                        help="ImageNet-C only, testing data corruption magnitude, choose from 1 to 5.")
    parser.add_argument('--vlm', type=str, choices=['albef16', 'blip16', 'clip16', 'clip32'],
                        help="Choose a VLM to finetune.")
    parser.add_argument('--batchsize', type=int, help="Batch size for fine-tuning.")
    parser.add_argument('--batchsize_test', type=int, help="Batch size for model evaluation.")
    parser.add_argument('--ckpt_path', type=str, help="Path for checkpoint (FLCP mode).")
    parser.add_argument('--json_path', type=str,
                        help="Path for json file which saves the prediction results.")
    parser.add_argument('--mode', type=str, choices=['lp', 'ft'],
                        help="Mode for fine-tuning.")
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--epoch', type=int, default=-1)
    parser.add_argument('--config_albef', default='./ALBEF/configs/ours.yaml')
    parser.add_argument('--config_blip', default='./BLIP/configs/ours.yaml')
    parser.add_argument('--albef_ori_ckpt', default='../my_ckpts/ALBEF.pth')
    # parser.add_argument('--output_dir', default='./ALBEF_output/')
    parser.add_argument('--albef_text_encoder', default='bert-base-uncased')
    # parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--distributed', default=False, type=bool)
    args = parser.parse_args()

    main(args)
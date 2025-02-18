import torch
import torch.nn.functional as F
import numpy as np
import random
from torch import optim
from datetime import datetime

import sys, os
sys.path.append(os.getcwd())

from utils.IN_dataset import *
from utils.lp_model import *
from my_util import *

import argparse

def main(args):
    local_rank_ = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank_)
    device = torch.device('cuda', local_rank_)
    torch.distributed.init_process_group(backend='nccl')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    current_time = datetime.now()
    time_string = current_time.strftime("%Y-%m-%d_%H:%M:%S")
    fn = time_string + "_" + args.vlm + "_" + args.mode + "_IN_e" \
         + str(args.epoch) + "_bs" + str(args.batchsize) + "*" + str(torch.cuda.device_count()) \
         + "_mGPUs_lr" + str(args.learning_rate)
    if local_rank_ == 0:
        if not os.path.exists(os.path.join("../my_ckpts", fn)):
            os.mkdir(os.path.join("../my_ckpts", fn))

    ff = open("../imagenet-classes.txt", 'r')
    in_class = []
    for l in ff.readlines():
        in_class.append(l[:-1].split(", ")[0])
    in_train_dl, in_val_dl, L, _ = get_IN_train_val(1000, args.batchsize, args.batchsize_test,
                                                    args.num_workers, mg=True)

    ft_whole_model = True if args.mode == 'ft' else False
    if args.vlm == "albef16":
        vlm_model, optimizer, fn, tokenizer = load_albef_model(args)
        model = LP_Net_A(vlm_model, 768, 1000, freeze_ab=not ft_whole_model).to(device)
    elif args.vlm == "blip16":
        vlm_model, optimizer, fn = load_blip_model(args)
        model = LP_Net_A(vlm_model, 768, 1000, freeze_ab=not ft_whole_model).to(device)
    elif "clip" in args.vlm:
        vlm_model, optimizer, fn = load_clip_model(
            args.epoch, args.batchsize, args.learning_rate,
            bb_num=int(args.vlm[-2:]), dataset_name=args.dataset, mode=args.mode
        )
        model = LP_Net(vlm_model, 512, 1000, freeze_clip=not ft_whole_model).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank_], output_device=local_rank_, find_unused_parameters=True
    )

    for epoch in range(args.epoch):
        if local_rank_ == 0:
            print("Epoch:", epoch)
        model.train()
        avg_train_loss = 0.0
        for iter, labeled_batch in enumerate(in_train_dl):
            optimizer.zero_grad()

            images, labels = labeled_batch
            images = images.to(device)
            labels = labels.to(device)
            logits_pred = model(images)
            loss = F.cross_entropy(logits_pred, labels)
            loss.backward()
            optimizer.step()

            """vis log of training"""
            end_t = datetime.now()
            if iter % args.log_freq == 0 and iter > 0 and local_rank_ == 0:
                print("iter:", iter, "/", len(in_train_dl),
                      "| losses:", round(loss.item(), 4),
                      "| ETA:", str(calculate_ETA(start_t, end_t, len(in_train_dl), iter, epoch, args.epoch)))
            start_t = datetime.now()
            avg_train_loss += loss.item()

        write_str = "epoch:" + str(epoch) + " loss:" + str(avg_train_loss / len(in_train_dl)) + "\n"
        if local_rank_ == 0:
            with open(os.path.join("../my_ckpts", fn, "log.txt"), 'a') as f:
                f.write(write_str)
                f.close()

        """Validation"""
        pred_list_all = []
        anno_list_all = []
        if (epoch + 1) % args.eval_epoch == 0:
            if local_rank_ == 0:
                print("Validation")
            model.eval()
            for iter, labeled_batch in enumerate(in_val_dl):
                images, labels = labeled_batch
                labels = labels.to(device)
                images = images.to(device)
                logits_pred = model(images)
                pred_list = torch.argmax(logits_pred, dim=1).cpu().tolist()
                anno_list = labels.cpu().tolist()
                pred_list_all += pred_list
                anno_list_all += anno_list
            pred_list_f = concat_all_gather(torch.tensor(pred_list_all).to(device))
            anno_list_f = concat_all_gather(torch.tensor(anno_list_all).to(device))
            R = sum(1 for x, y in zip(pred_list_f.cpu().tolist(), anno_list_f.cpu().tolist()) if x == y)
            if local_rank_ == 0:
                print("acc:", float(R) / anno_list_f.shape[0])
                write_str = "epoch:" + str(epoch) + " acc:" + str(float(R) / anno_list_f.shape[0]) + "\n"
                with open(os.path.join("../my_ckpts", fn, "log.txt"), 'a') as f:
                    f.write(write_str)
                    f.close()

        """Saving ckpts"""
        if local_rank_ == 0:
            if (epoch + 1) % args.ckpt_epoch == 0:
                if not os.path.exists(os.path.join("../my_ckpts", fn)):
                    os.mkdir(os.path.join("../my_ckpts", fn))
                torch.save(model.state_dict(), os.path.join("../my_ckpts", fn, str(epoch).zfill(5) + ".pyth"))
                if local_rank_ == 0:
                    print("Saved checkpoints for epoch", epoch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--vlm', type=str, choices=['albef16', 'blip16', 'clip16', 'clip32'],
                        help="Choose a VLM to finetune.")
    parser.add_argument('--batchsize', type=int, help="Batch size for fine-tuning.")
    parser.add_argument('--batchsize_test', type=int, help="Batch size for model evaluation.")
    parser.add_argument('--epoch', type=int, help="Number of epochs for fine-tuning.")
    parser.add_argument('--eval_epoch', type=int, help="Epoch interval for evaluation.")
    parser.add_argument('--ckpt_epoch', type=int, help="Epoch interval for checkpoints.")
    parser.add_argument('--learning_rate', type=float, help="Learning rate for fine-tuning.")
    parser.add_argument('--log_freq', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--contrastive_temp', type=float, default=0.07)
    parser.add_argument('--mode', type=str, choices=['lp', 'ft'],
                        help="Mode for fine-tuning.")
    parser.add_argument('--dataset', type=str, default='IN')
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
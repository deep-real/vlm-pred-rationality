import torch
import torch.nn.functional as F
from datetime import datetime

import argparse
try:
    import ruamel_yaml as yaml
except:
    import ruamel.yaml as yaml

import sys, os
sys.path.append(os.getcwd())

from my_util import *
from utils.lp_model import *

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
            args.epoch, args.batchsize, args.learning_rate,
            bb_num=int(args.vlm[-2:]), dataset_name=args.dataset, mode=args.mode
        )
        model = LP_Net(vlm_model, 512, cls_num, freeze_clip=not ft_whole_model).to(device)
    else:
        raise NotImplementedError
    for epoch in range(args.epoch):
        print("Epoch:", epoch)
        model.train()
        avg_train_loss = 0.0
        for iter, labeled_batch in enumerate(data_train_dl):
            optimizer.zero_grad()

            if args.dataset == "cub":
                images, labels = labeled_batch
            else:
                images, labels, _, _ = labeled_batch
            images = images.to(device)
            labels = labels.to(device)

            logits_pred = model(images)
            loss = F.cross_entropy(logits_pred, labels)

            loss.backward()
            optimizer.step()

            """vis log of training"""
            end_t = datetime.now()
            if iter % args.log_freq == 0 and iter > 0:
                print("iter:", iter, "/", len(data_train_dl),
                      "| losses:", round(loss.item(), 4),
                      "| ETA:", str(calculate_ETA(start_t, end_t, len(data_train_dl), iter, epoch, args.epoch)))
            start_t = datetime.now()
            avg_train_loss += loss.item()

        write_str = "epoch:" + str(epoch) + " loss:" + str(avg_train_loss / len(data_train_dl)) + "\n"
        with open(os.path.join("../my_ckpts", fn, "log.txt"), 'a') as f:
            f.write(write_str)
            f.close()

        """Validation"""
        if (epoch + 1) % args.eval_epoch == 0:
            print("Validation")
            model.eval()
            R = 0
            for iter, labeled_batch in enumerate(data_test_dl):
                if args.dataset == "cub":
                    images, labels = labeled_batch
                else:
                    images, labels, _, _ = labeled_batch
                images = images.to(device)
                labels = labels.to(device)
                logits_pred = model(images)
                pred_list = torch.argmax(logits_pred, dim=1).cpu().tolist()
                anno_list = labels.cpu().tolist()
                R += sum(1 for x, y in zip(pred_list, anno_list) if x == y)
            print("acc:", float(R) / L)
            write_str = "epoch:" + str(epoch) + " acc:" + str(float(R) / L) + "\n"
            with open(os.path.join("../my_ckpts", fn, "log.txt"), 'a') as f:
                f.write(write_str)
                f.close()

        """Saving ckpts"""
        if (epoch + 1) % args.ckpt_epoch == 0:
            if not os.path.exists(os.path.join("../my_ckpts", fn)):
                os.mkdir(os.path.join("../my_ckpts", fn))
            torch.save(model.state_dict(), os.path.join("../my_ckpts", fn, str(epoch).zfill(5) + ".pyth"))
            print("Saved checkpoints for epoch", epoch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--dataset', type=str, choices=['cub', 'sd', 'ct'],
                        help="Dataset options; cub: CUB-200-2011, sd: Stanford-Dogs, caltech: CalTech101.")
    parser.add_argument('--vlm', type=str, choices=['albef16', 'blip16', 'clip16', 'clip32'],
                        help="Choose a VLM to finetune.")
    parser.add_argument('--batchsize', type=int, help="Batch size for fine-tuning.")
    parser.add_argument('--batchsize_test', type=int, help="Batch size for model evaluation.")
    parser.add_argument('--learning_rate', type=float, help="Learning rate for fine-tuning.")
    parser.add_argument('--epoch', type=int, help="Number of epochs for fine-tuning.")
    parser.add_argument('--eval_epoch', type=int, help="Epoch interval for evaluation.")
    parser.add_argument('--ckpt_epoch', type=int, help="Epoch interval for checkpoints.")
    parser.add_argument('--log_freq', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--mode', type=str, choices=['lp', 'ft'],
                        help="Mode for fine-tuning.")
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
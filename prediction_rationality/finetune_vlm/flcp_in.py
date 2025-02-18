import torch
import torch.nn.functional as F
import numpy as np
import random
from torch import optim
from datetime import datetime

import sys, os
sys.path.append(os.getcwd())

from utils.IN_dataset import *
import clip
from my_util import *
import argparse

def main(args):
    local_rank_ = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank_)
    device = torch.device('cuda', local_rank_)
    torch.distributed.init_process_group(backend='nccl')
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    current_time = datetime.now()
    time_string = current_time.strftime("%Y-%m-%d_%H:%M:%S")
    fn = time_string + "_" + args.vlm + "_FLCP_IN_e" \
         + str(args.epoch) + "_bs" + str(args.batchsize) \
         + "*" + str(torch.cuda.device_count()) + "_mGPUs_lr" + str(args.learning_rate)
    if local_rank_ == 0:
        if not os.path.exists(os.path.join("../my_ckpts", fn)):
            os.mkdir(os.path.join("../my_ckpts", fn))

    ff = open("../datasets/IN/imagenet-classes.txt", 'r')
    in_class = []
    for l in ff.readlines():
        in_class.append(l[:-1].split(", ")[0])
    in_train_dl, in_val_dl, L, _ = get_IN_train_val(
        1000, args.batchsize, args.batchsize_test, args.num_workers, mg=True
    )

    if args.vlm == "albef16":
        model, optimizer, _, tokenizer = load_albef_model(args)
    elif args.vlm == "blip16":
        model, optimizer, _ = load_blip_model(args)
    elif "clip" in args.vlm:
        model, preprocess = clip.load(f"ViT-B/{int(args.vlm[-2:])}", device=device, jit=False)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank_], output_device=local_rank_, find_unused_parameters=True
        )
    else:
        raise NotImplementedError
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate,
                           betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)

    for epoch in range(args.epoch):
        if local_rank_ == 0:
            print("Epoch:", epoch)
        model.train()

        avg_train_loss = 0.0
        for iter, labeled_batch in enumerate(in_train_dl):
            optimizer.zero_grad()
            images, labels = labeled_batch
            images = images.to(device)
            if 'clip' in args.vlm:
                texts = label2text2embedding(labels, in_class, device)
                texts = texts.to(device)
            elif args.vlm == "albef16":
                texts = label2text2embedding_albef(labels, in_class, device, tokenizer)
                texts = texts.to(device)
            elif args.vlm == "blip16":
                texts = label2text_blip(labels, in_class)

            if 'clip' in args.vlm:
                image_features = model.module.encode_image(images)
                text_features = model.module.encode_text(texts)
                image_features_norm = F.normalize(image_features, dim=-1)
                text_features_norm = F.normalize(text_features, dim=-1)
                sim_i2t = image_features_norm @ text_features_norm.T / args.contrastive_temp
                sim_t2i = text_features_norm @ image_features_norm.T / args.contrastive_temp
                ground_truth = torch.eye(image_features.shape[0]).to(device)
                total_loss_1 = -torch.sum(F.log_softmax(sim_i2t, dim=1) * ground_truth, dim=1).mean()
                total_loss_2 = -torch.sum(F.log_softmax(sim_t2i, dim=1) * ground_truth, dim=1).mean()
                loss = (total_loss_1 + total_loss_2) / 2
                loss.backward()
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)
            else:
                # blip
                loss, _ = model(images, texts)
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
        # print(write_str)
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
                images = images.to(device)
                texts = [f"a photo of a {c}" for c in in_class]
                if 'clip' in args.vlm:
                    texts = text2embedding(texts, device)
                    with torch.no_grad():
                        image_feats = model.module.encode_image(images)
                        image_embeds = F.normalize(image_feats, dim=-1)
                        text_outputs = model.module.encode_text(texts)
                        text_embeds = F.normalize(text_outputs, dim=-1)
                elif 'albef16' == args.vlm:
                    with torch.no_grad():
                        image_feats = model.visual_encoder(images)
                        image_embeds = model.vision_proj(image_feats[:, 0, :])
                        image_embeds = F.normalize(image_embeds, dim=-1)
                        text_inputs = tokenizer(texts, padding='max_length', truncation=True, max_length=30,
                                                return_tensors="pt").to(device)
                        text_outputs = model.text_encoder(text_inputs.input_ids,
                                                                 attention_mask=text_inputs.attention_mask,
                                                                 mode='text')
                        text_feats = text_outputs.last_hidden_state
                        text_embeds = F.normalize(model.text_proj(text_feats[:, 0, :]))
                else:
                    with torch.no_grad():
                        image_feats = model.visual_encoder(images)
                        image_embeds = model.vision_proj(image_feats[:, 0, :])
                        image_embeds = F.normalize(image_embeds, dim=-1)
                        text_inputs = model.tokenizer(
                            texts, padding='max_length', truncation=True, max_length=35, return_tensors="pt"
                        ).to(device)
                        text_outputs = model.text_encoder(text_inputs.input_ids,
                                                          attention_mask=text_inputs.attention_mask,
                                                          mode='text')
                        text_feats = text_outputs.last_hidden_state
                        text_embeds = F.normalize(model.text_proj(text_feats[:, 0, :]))
                similarity = (100.0 * image_embeds @ text_embeds.T).softmax(dim=-1)
                values, indices = similarity.topk(k=1, dim=1)
                pred_list = indices.squeeze().cpu().tolist()
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
    parser.add_argument('--mode', type=str, default='flcp')
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
    # parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    main(args)
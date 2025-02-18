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
sys.path.append(os.path.join(os.getcwd(), 'utils'))
sys.path.append(os.path.abspath('/usa/wqtwjt/CLIP/'))
# sys.path.append(os.path.abspath('/usa/wqtwjt/CLIP/prediction_rationality/utils'))
# sys.path.append(os.path.abspath('/usa/wqtwjt/CLIP/prediction_rationality/utils/ALBEF'))
# sys.path.append(os.path.join(os.getcwd(), 'prediction_rationality', 'utils'))
print(sys.path)
from my_util import *

import clip

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model
    if args.vlm == "albef16":
        model, optimizer, fn, tokenizer = load_albef_model(args)
    elif args.vlm == "blip16":
        model, optimizer, fn = load_blip_model(args)
    elif "clip" in args.vlm:
        model, optimizer, fn = load_clip_model(
            args.epoch, args.batchsize, args.learning_rate,
            bb_num=int(args.vlm[-2:]), dataset_name=args.dataset, mode=args.mode
        )
    else:
        raise NotImplementedError
    # dataset
    if args.dataset == "cub":
        data_train_dl, data_test_dl, L, class_list = load_cub_dataset(
            args.batchsize, args.batchsize_test, args.num_workers
        )
    elif args.dataset == "sd":
        data_train_dl, data_test_dl, L, class_list = load_sd_dataset(
            args.batchsize, args.batchsize_test, args.num_workers
        )
    elif args.dataset == "ct":
        data_train_dl, data_test_dl, L, class_list = load_ct_dataset(
            args.batchsize, args.batchsize_test, args.num_workers
        )
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
            if 'clip' in args.vlm:
                texts = label2text2embedding(labels, class_list, device)
                texts = texts.to(device)
            elif args.vlm == "albef16":
                texts = label2text2embedding_albef(labels, class_list, device, tokenizer)
                texts = texts.to(device)
            elif args.vlm == "blip16":
                texts = label2text_blip(labels, class_list)

            if 'clip' in args.vlm:
                image_features = model.encode_image(images)
                text_features = model.encode_text(texts)
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
                loss, _ = model(images, texts)
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
            if 'clip' in args.vlm:
                pred_list = []
                anno_list = []
                text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in class_list]).to(device)
                for iter, labeled_batch in enumerate(data_test_dl):
                    if args.dataset == "cub":
                        images, labels = labeled_batch
                    else:
                        images, labels, _, _ = labeled_batch
                    images = images.to(device)
                    labels = labels.to(device)
                    # Calculate features
                    with torch.no_grad():
                        image_features = model.encode_image(images)
                        text_features = model.encode_text(text_inputs)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                    values, indices = similarity.topk(1, dim=1)
                    R += sum(1 for x, y in zip(indices.squeeze().cpu().tolist(), labels.cpu().tolist()) if x == y)
                    pred_list += indices.squeeze().cpu().tolist()
                    anno_list += labels.cpu().tolist()
            else:
                for iter, labeled_batch in enumerate(data_test_dl):
                    if args.dataset == "cub":
                        images, labels = labeled_batch
                    else:
                        images, labels, _, _ = labeled_batch
                    images = images.to(device)
                    texts = [f"a photo of a {c}" for c in class_list]
                    with torch.no_grad():
                        image_feats = model.visual_encoder(images)
                        image_embeds = model.vision_proj(image_feats[:, 0, :])
                        image_embeds = F.normalize(image_embeds, dim=-1)
                        if 'albef16' == args.vlm:
                            text_inputs = tokenizer(texts, padding='max_length',
                                                    truncation=True, max_length=30,
                                                    return_tensors="pt").to(device)
                        else:
                            text_inputs = model.tokenizer(texts, padding='max_length',
                                                          truncation=True, max_length=30,
                                                          return_tensors="pt").to(device)
                        text_outputs = model.text_encoder(text_inputs.input_ids,
                                                          attention_mask=text_inputs.attention_mask, mode='text')
                        text_feats = text_outputs.last_hidden_state
                        text_embeds = F.normalize(model.text_proj(text_feats[:, 0, :]))
                    similarity = (100.0 * image_embeds @ text_embeds.T).softmax(dim=-1)
                    values, indices = similarity.topk(k=1, dim=1)
                    pred_list = indices.squeeze().cpu().tolist()
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
                        help="Dataset options; cub: CUB-200-2011, sd: Stanford-Dogs, ct: CalTech101.")
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
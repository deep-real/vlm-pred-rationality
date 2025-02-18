# Beyond Accuracy: On the Effects of Fine-tuning Towards Vision-Language Model's Prediction Rationality (AAAI-2025)

This repository holds the Pytorch implementation of "Beyond Accuracy: On the Effects of Fine-tuning Towards Vision-Language Model's Prediction Rationality" by [Qitong Wang](https://wqtwjt1996.github.io/), [Tang Li](https://tangli0305.github.io/), [Kien X. Nguyen](https://nyquixt.github.io/profile/), and [Xi Peng](https://deep-real.github.io/dr_xipeng.html). If you find our code or paper useful in your research, please consider citing:

```
@InProceedings{Wang_2025_Rationale,
 author = {Wang, Qitong and Li, Tang and Nguyen, Kien X. and Peng, Xi},
 title = {Beyond Accuracy: On the Effects of Fine-tuning Towards Vision-Language Model's Prediction Rationality},
 booktitle = {In Proceedings of the Association for the Advancement of Artificial Intelligence (AAAI)},
 month = {February},
 year = {2025},
}
```

To run our code or reimplement our results in the main paper successfully, please read this carefully and follow the steps below.

## Step-1: Virtual Environment Installation.
```
pip install -r requirements.txt
```
Note that your virtual environment configuration, such as the CUDA version and GPU type, may differ from ours. Therefore, please also consider referring to the PyTorch official website for your environment setup.

## Step-2: Dataset Installation and Pretrained Weights Preparation.

### CalTech-101

Visit https://data.caltech.edu/records/mzrjq-6wc02 for data downloading. The downloaded and extracted files should be placed in the './datasets/caltech_101/' directory.

### CUB-200-2011

Visit https://data.caltech.edu/records/65de6-vp158 for images and annotations, and visit https://data.caltech.edu/records/w9d68-gec53 for segmentation masks.
The two folders "CUB_200_2011" and "segmentations" should be placed in the './datasets/CUB/' directory.

### ImageNet-1K

Run the following command to go to the "datasets" folder, then make new folders called "IN" and "ilsvrc":
```
cd datasets
mkdir IN
cd IN
mkdir ilsvrc
```
Visit https://image-net.org/index.php for ILSVRC-2012 data downloading (training and validation data). The folder structure should be as follows:
```
datasets/IN/ilsvrc
├── train
│   ├── n01440764
│   ├── n01443537
│   └── ...
└── val
    ├── n01440764
    ├── n01443537
    └── ...
```
where folder names such as "n01440764", and "n01443537" inside each dataset split folder denote class ID. 

### Stanford-Dogs

Visit http://vision.stanford.edu/aditya86/ImageNetDogs/ for data downloading. Download "Images (757MB)" and "Annotations (21MB)" only. The downloaded and extracted 'Images' and 'Annotation' folders should be placed in the './datasets/Stanford_Dogs/' directory.

### ImageNet-C

Run the following command to go to the "datasets" folder, then make a new folder called "ImageNet-C":
```
cd datasets/IN
mkdir ImageNet-C
```
Visit https://zenodo.org/records/2235448 for data downloading. The downloaded and extracted folders should be placed in the './datasets/IN/ImageNet-C/' directory. The folder structure should be as follows:
```
datasets/IN/ImageNet-C
├── brightness
│   ├── 1
│   ├── 2
│   ├── 3
│   ├── 4
│   └── 5
├── contrast
│   ├── 1
│   └── ...
├── defocus_blur
│   └── ...
├── ...
└── zoom_blur
    └── ...
```
where folder names such as "brightness", and "contrast" denote corruption type, and folder names such as "1", and "2" inside each corruption type folder denote corruption strength. 
In the corruption strength folders (such as "./datasets/IN/ImageNet-C/zoom_blur/1"), the structure should be the same as "./datasets/IN/ilsvrc/val".

For pre-trained weights of ALBEF-ViT-B16 and BLIP-ViT-B16, please visit "https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/ALBEF.pth" and "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth" to download, then put them in the ./my_ckpts/ folder in the root folder.
```
mkdir my_ckpts
wget https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/ALBEF.pth
wget https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth
```
Lastly, please rename "model_base.pth" as "BLIP.pth".

## Step-3: Model Fine-tuning.

From now on (Step-3 to 6), here are some important notes which ensure you understand the instructions provided below.

1. Dataset Abbreviations: CT: CalTech-101, CUB: CUB-200-2011, IN: ImageNet-1K, SD: Stanford-Dogs, IN-c: ImageNet-C.
2. Method Abbreviations: ZS: Zero-Shot, LP: Linear-Probing, FLCP: Finetune like CLIP Pretrain, FT: Fine-tuning.
3. Before running one specific Python file in each step, please edit the variables from the argparse. It also provides detailed descriptions of all the variables, so please consider reading these carefully.
4. For the detailed hyperparameter settings of our implementation, please check the Sect. "Experimental Setup" from the main paper and Sect. "More VLM Finetuning Details" from supplementary material.

Run the following command to go to the "prediction_rationality" folder:
```
cd prediction_rationality
```
Then use the "finetune_vlm/flcp.py" and "finetune_vlm/lp_ft.py" to finetune your model (except the ImageNet-1K dataset, which uses "finetune_vlm/flcp_in.py" and "finetune_vlm/lp_ft_in.py").
Here we give two examples:
1. fine-tuning with the FLCP method using the ALBEF model on the CUB dataset:
```
CUDA_VISIBLE_DEVICES=0 python -m finetune_vlm.flcp --dataset cub --vlm albef16 --batchsize 32 --batchsize_test 32 --epoch 10 --eval_epoch 1 --ckpt_epoch 5 --learning_rate 0.00001
```
2. fine-tuning with the LP method using the CLIP-ViT-B-16 model on the SD dataset:
```
CUDA_VISIBLE_DEVICES=0 python -m finetune_vlm.lp_ft --mode lp --dataset sd --vlm clip16 --batchsize 32 --batchsize_test 32 --epoch 10 --eval_epoch 1 --ckpt_epoch 5 --learning_rate 0.001
```
Note that we fine-tune using the ImageNet-1K dataset with multiple GPUs, here is another example of a command if you want to train with multiple GPUs:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 23373 --use_env finetune_vlm/flcp_in.py --vlm blip16 --batchsize 64 --batchsize_test 64 --epoch 10 --eval_epoch 1 --ckpt_epoch 5 --learning_rate 0.00001
```

## Step-4: Model Evaluation of Prediction Accuracy.
Using "test_pred_vlm/zs_flcp.py" and "test_pred_vlm/lp_ft.py" to test the model's prediction accuracy.
Here we give four examples:
1. Testing with the ZS method using the CLIP-ViT-B16 model on the CT dataset:
```
CUDA_VISIBLE_DEVICES=0 python -m test_pred_vlm.zs_flcp --mode zs --dataset ct --vlm clip16 --batchsize 64 --batchsize_test 64 --json_path {PATH_TO_JSON_FILE_FOR_PREDICTIONS}
```
2. Testing with the LP method using the BLIP model on the IN dataset:
```
CUDA_VISIBLE_DEVICES=0 python -m test_pred_vlm.lp_ft --mode lp --dataset in --vlm blip16 --batchsize 64 --batchsize_test 64 --json_path {PATH_TO_JSON_FILE_FOR_PREDICTIONS} --ckpt_path ../my_ckpts/{PATH_TO_YOUR_CKPT}
```
3. Testing with the FLCP method using the CLIP-ViT-B32 model on the IN-C dataset (with corruption type: zoom_blur, and corruption magnitude: 3):
```
CUDA_VISIBLE_DEVICES=0 python -m test_pred_vlm.zs_flcp --mode flcp --dataset in-c --corrupt_name zoom_blur --corrupt_magnitude 3 --vlm clip32 --batchsize 64 --batchsize_test 64 --json_path {PATH_TO_JSON_FILE_FOR_PREDICTIONS} --ckpt_path ../my_ckpts/{PATH_TO_YOUR_CKPT}
```
4. Testing with the FT method using the ALBEF model on the CUB dataset:
```
CUDA_VISIBLE_DEVICES=0 python -m test_pred_vlm.lp_ft --mode ft --dataset cub --vlm albef16 --batchsize 64 --batchsize_test 64 --json_path {PATH_TO_JSON_FILE_FOR_PREDICTIONS} --ckpt_path ../my_ckpts/{PATH_TO_YOUR_CKPT}
```

## Step-5: Model Evaluation of Rationale.
Python files for these steps are in the "test_rationale_vlm" folder, which is named in this format: "clip_{METHOD_NAME}_{DATASET_NAME}" (such as "clip_zs_flcp_cub", "blip_lp_ft_in").
Here we give four examples:
1. Testing rationale with the ZS method using the CLIP-ViT-B-16 model on the SD dataset:
```
CUDA_VISIBLE_DEVICES=0 python -m test_rationale_vlm.clip_zs_flcp_sd --mode zs --clip_bb_num 16 --batchsize 64 --batchsize_test 64 --pred_json_path {PATH_TO_JSON_FILE_FOR_PREDICTIONS} --rationale_json_path {PATH_TO_JSON_FILE_FOR_RATIOINLES} --ckpt_path ../my_ckpts/{PATH_TO_YOUR_CKPT}
```
2. Testing rationale with LP method using the CLIP-ViT-B-16 model on the CT dataset:
```
CUDA_VISIBLE_DEVICES=0 python -m test_rationale_vlm.clip_lp_ft_ct --mode lp --clip_bb_num 16 --batchsize 64 --batchsize_test 64 --pred_json_path {PATH_TO_JSON_FILE_FOR_PREDICTIONS} --rationale_json_path {PATH_TO_JSON_FILE_FOR_RATIOINLES} --ckpt_path ../my_ckpts/{PATH_TO_YOUR_CKPT}
```
3. Testing rationale with the FLCP method using the CLIP-ViT-B-32 model on the IN dataset:
```
CUDA_VISIBLE_DEVICES=0 python -m test_rationale_vlm.clip_zs_flcp_in --mode flcp --clip_bb_num 32 --dataset in --batchsize 64 --batchsize_test 64 --pred_json_path {PATH_TO_JSON_FILE_FOR_PREDICTIONS} --rationale_json_path {PATH_TO_JSON_FILE_FOR_RATIOINLES} --ckpt_path ../my_ckpts/{PATH_TO_YOUR_CKPT}
```
4. Testing rationale with LP method using the CLIP-ViT-B-32 model on the IN-C dataset (with corruption type: frost, and corruption magnitude: 5):
```
CUDA_VISIBLE_DEVICES=0 python -m test_rationale_vlm.clip_zs_flcp_in --mode ft --clip_bb_num 32 --dataset in-c --corrupt_name frost --corrupt_magnitude 5 --batchsize 64 --batchsize_test 64 --pred_json_path {PATH_TO_JSON_FILE_FOR_PREDICTIONS} --rationale_json_path {PATH_TO_JSON_FILE_FOR_RATIOINLES} --ckpt_path ../my_ckpts/{PATH_TO_YOUR_CKPT}
```
Please check the configuration code in the argparse for more details.

## Step-6: PT, and IR Evaluations.
Run this command:
```
python PT_IR_analysis.py
```
Note that prediction annotations JSON files (for "--anno_json_path" in the argparse) are in the "./datasets/annos" folder, named "ANNO_{DATASET_NAME}.json" (such as "ANNO_CUB.json" and "ANNO_IN.json"). 
For the evaluation on ImageNet-C, please use "ANNO_IN.json".

Codebase References:

https://github.com/salesforce/ALBEF.

https://github.com/salesforce/BLIP.

https://github.com/openai/CLIP.

https://github.com/hila-chefer/Transformer-MM-Explainability.

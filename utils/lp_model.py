import torch
import torch.nn as nn
import torch.nn.functional as F

class LP_Net(nn.Module):
    def __init__(self, clip_model, dim_in, num_classes, freeze_clip=True):
        super(LP_Net, self).__init__()

        self.clip_feature_extractor = clip_model
        self.classifier = nn.Linear(in_features=dim_in, out_features=num_classes)
        if freeze_clip:
            for param in self.clip_feature_extractor.parameters():
                param.requires_grad = False
        for param in self.parameters():
            if param.dtype == torch.float16:
                param.data = param.data.to(torch.float32)

    def forward(self, x):
        z = self.clip_feature_extractor.encode_image(x)
        pred = self.classifier(z)
        return pred

class LP_Net_A(nn.Module):
    def __init__(self, ab_model, dim_in, num_classes, freeze_ab=True):
        super(LP_Net_A, self).__init__()

        self.ab_feature_extractor = ab_model.visual_encoder
        self.classifier = nn.Linear(in_features=dim_in, out_features=num_classes)
        if freeze_ab:
            for param in self.ab_feature_extractor.parameters():
                param.requires_grad = False

    def forward(self, x, exp_mode=False):
        z = self.ab_feature_extractor(x, exp_mode)
        pred = self.classifier(z[:, 0, :])
        return pred
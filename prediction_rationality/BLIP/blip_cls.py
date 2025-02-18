from BLIP.med import BertConfig, BertModel
from transformers import BertTokenizer

import torch
from torch import nn
import torch.nn.functional as F

from BLIP.blip import create_vit, init_tokenizer, load_checkpoint

class BLIP(nn.Module):
    def __init__(self,                 
                 med_config = 'configs/med_config.json',  
                 image_size = 224,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,                      
                 embed_dim = 256,
                 negative_all_rank = False,
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """               
        super().__init__()
        
        self.visual_encoder, vision_width = create_vit(vit,image_size, vit_grad_ckpt, vit_ckpt_layer)
        self.tokenizer = init_tokenizer()   
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)          

        text_width = self.text_encoder.config.hidden_size
        
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.temp = nn.Parameter(0.07*torch.ones([]))
        self.negative_all_rank = negative_all_rank
        
        
    def forward(self, image, caption):
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)
        
        image_embeds = self.visual_encoder(image)
        image_feat = F.normalize(self.vision_proj(image_embeds[:,0,:]),dim=-1)    
        
        text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=35, 
                              return_tensors="pt").to(image.device) 
        
        text_output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask,                      
                                        return_dict = True, mode = 'text')            
        text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:,0,:]),dim=-1)

        sim_i2t = image_feat @ text_feat.T / self.temp
        sim_t2i = text_feat @ image_feat.T / self.temp

        sim_targets = torch.zeros(sim_i2t.size()).to(image.device)
        sim_targets.fill_diagonal_(1)

        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_targets, dim=1).mean()

        loss_ita = (loss_i2t+loss_t2i)/2

        return loss_ita, None


def blip_cls(pretrained='',**kwargs):
    model = BLIP(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained)
        # print("missing keys:")
        # print(msg.missing_keys)
    return model 


@torch.no_grad()
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


class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        torch.distributed.all_reduce(all_gradients)
        return all_gradients[torch.distributed.get_rank()]


def all_gather_with_grad(tensors):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    """
    # Queue the gathered tensors
    world_size = torch.distributed.get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors

    tensor_all = GatherLayer.apply(tensors)

    return torch.cat(tensor_all, dim=0)

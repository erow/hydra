# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.distributed
import torch.nn as nn
from .filter import Filter, ConvFilter
import gin

@gin.configurable(denylist=['dim','mlp_dim','T'])
class MoCo(nn.Module):
    """
    Build a MoCo model with a base encoder, a momentum encoder, and two MLPs
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, 
                 dim=512, mlp_dim=4096, T=1.0, 
                 alpha=0, beta=0.0, compile=False,
                 norm='ln-none',
                 num_layers = 3,
                 warmup = 10,
                 clip=False,
                 num_classes=512):
        """
        dim: feature dimension (default: 512 the same as the caption embedding)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        """
        super(MoCo, self).__init__()

        self.T = T
        self.alpha=alpha
        self.beta = beta
        self.num_classes=num_classes
        self.clip=clip
        self.num_layers = num_layers 
        # build encoders
        self.base_encoder = base_encoder(num_classes=mlp_dim)
        self.momentum_encoder = base_encoder(num_classes=mlp_dim)
        self.filter = Filter(self.num_classes,dim)
        self.norm = norm
        self.warmup = warmup

        self._build_projector_and_predictor_mlps(dim, mlp_dim)
        self.scale_logit = nn.Parameter(torch.zeros(1)-np.log(T))

        
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient
        
        if compile:
            self.base_encoder = torch.compile(self.base_encoder)
            self.momentum_encoder = torch.compile(self.momentum_encoder)
            self.filter = torch.compile(self.filter)
            self.predictor = torch.compile(self.predictor)
    
    @torch.no_grad()
    def representation(self, x):
        return self.momentum_encoder(x)
    
    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_norm='bn'):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))

        if last_norm=='bn':
            # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
            # for simplicity, we further removed gamma in BN
            mlp.append(nn.BatchNorm1d(output_dim, affine=False))
        elif last_norm=='ln':
            # BN will prevent gate close
            mlp.append(nn.LayerNorm(output_dim))
        elif last_norm=='none':
            pass
        else:
            raise ValueError(f'last_norm={last_norm} not supported')

        return nn.Sequential(*mlp)

    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        pass

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def contrastive_loss(self, q, k):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        # gather all targets
        k = concat_all_gather(k)
        # Einstein sum is more intuitive
        scale =  self.scale_logit.exp()
        logits = torch.einsum('nc,mc->nm', [q, k]) * scale
        N = logits.shape[0]  # batch size per GPU
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        labels = (torch.arange(N, dtype=torch.long) + N * rank).cuda()
        return nn.CrossEntropyLoss()(logits, labels)


    def disparate_loss(self, z1, k2, y1, posy):
        k2 = concat_all_gather(k2)
        fz1,fz2 = self.filter(z1, k2, y1,posy)
        
        scale = self.scale_logit.exp()
        logits = scale * self.filter.contrast(fz1,fz2)
        N = z1.shape[0]
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        labels = (torch.arange(N, dtype=torch.long) + N * rank).cuda()
        return nn.CrossEntropyLoss()(logits, labels)
        

    def forward(self, x1, x2, m, targets,epoch):
        """
        Input:
            x1: first views of images
            x2: second views of images
            m: moco momentum
        Output:
            loss
        """

        self.log = {}
        # shuffle trick, compose a positive pair from a random sample of the batch
        shuffle_idx = torch.randperm(len(x1)).to(x1.device)
        y = targets.clone()
        sy = targets[shuffle_idx].contiguous()
        
        # compute features
        z1 = self.base_encoder(x1)
        z2 = self.base_encoder(x2)
        q1 = self.predictor(z1)
        q2 = self.predictor(z2)

        if self.clip:
            loss = (self.contrastive_loss(q1, y) + self.contrastive_loss(q2, y))/2

        else:
            with torch.no_grad():  # no gradient
                self._update_momentum_encoder(m)  # update the momentum encoder

                # compute momentum features as targets
                k1 = self.momentum_encoder(x1).contiguous()
                k2 = self.momentum_encoder(x2).contiguous()

            # disparate contrast
            loss = (
                self.disparate_loss(q1,k2,y,sy) + 
                self.disparate_loss(q2,k1,y,sy))/2

       
        with torch.no_grad():
            # activation, entropy = self.filter.gate.statistics()
            # self.log['activation'] = activation.item()
            # self.log['entropy'] = entropy.item()
            self.log['scale'] = self.scale_logit.exp().item()
            self.log['z@sim'] = nn.functional.cosine_similarity(z1,z2).mean().item()
        return loss, self.log
    


    

class MoCo_ResNet(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.fc.weight.shape[1]
        del self.base_encoder.fc, self.momentum_encoder.fc # remove original fc layer

        norm1,norm2 = self.norm.split('-') # bn-none for resnet in MoCo
        # projectors
        self.base_encoder.fc = self._build_mlp(self.num_layers, hidden_dim, mlp_dim, dim, norm1)
        self.momentum_encoder.fc = self._build_mlp(self.num_layers, hidden_dim, mlp_dim, dim, norm1)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim, norm2)


class MoCo_ViT(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.head.weight.shape[1]
        del self.base_encoder.head, self.momentum_encoder.head # remove original fc layer

        norm1,norm2 = self.norm.split('-') # bn-bn for resnet in MoCo
        # projectors
        self.base_encoder.head = self._build_mlp(self.num_layers, hidden_dim, mlp_dim, dim, norm1)
        self.momentum_encoder.head = self._build_mlp(self.num_layers, hidden_dim, mlp_dim, dim, norm1)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim, norm2)


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if not torch.distributed.is_initialized():
        return tensor
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def multipos_ce_loss(logits, pos_mask,exclude_mask=None):
    if exclude_mask is None:
        exclude_mask = pos_mask
    logits = logits - logits.mean(1,keepdim=True)
    similarity = logits.exp()
    N = similarity.size(0)
 
    # InfoNCE loss 
    ## exclude the positives and class pairs
    neg = (similarity*(~exclude_mask)).sum(1,keepdim=True)
    loss = torch.sum(pos_mask* (torch.log(similarity + neg) - logits))/pos_mask.sum()
    loss = loss.mean()
   
    return loss
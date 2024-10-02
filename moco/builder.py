# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from .filter import Filter, ConvFilter

class MoCo(nn.Module):
    """
    Build a MoCo model with a base encoder, a momentum encoder, and two MLPs
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=256, mlp_dim=4096, T=1.0):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        """
        super(MoCo, self).__init__()

        self.T = T
        self.num_classes=1000
        # build encoders
        self.base_encoder = base_encoder(num_classes=mlp_dim)
        self.momentum_encoder = base_encoder(num_classes=mlp_dim)
        self.filter = ConvFilter(self.num_classes,dim)

        self._build_projector_and_predictor_mlps(dim, mlp_dim)

        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                # mlp.append(nn.BatchNorm1d(dim2, affine=False))
                # BN will prevent gate closing
                mlp.append(nn.LayerNorm(output_dim)) 

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
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        N = logits.shape[0]  # batch size per GPU
        labels = (torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()).cuda()
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)

    def forward(self, x1, x2, m,targets):
        """
        Input:
            x1: first views of images
            x2: second views of images
            m: moco momentum
        Output:
            loss
        """

        self.log = {}
        # compute features
        q1 = self.predictor(self.base_encoder(x1))
        q2 = self.predictor(self.base_encoder(x2))
        with torch.no_grad():  # no gradient
            self._update_momentum_encoder(m)  # update the momentum encoder

            # compute momentum features as targets
            k1 = self.momentum_encoder(x1)
            k2 = self.momentum_encoder(x2)

        instance_loss =  self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)

        # shuffle trick
        shuffle_idx = torch.randperm(len(k1)).to(k1.device)
        sk1 = k1[shuffle_idx]
        sk2 = k2[shuffle_idx]
        y1 = targets.clone()
        sy = targets[shuffle_idx]
        # disparate contrast
        disparate_loss = self.disparate_loss(q1,sk1,x1,x2) + self.disparate_loss(q2,sk2,x1,x2)
        #
        # class_loss = self.class_loss(q1,sk1,y1,sy) + self.class_loss(q2,sk2,y1,sy)
        
        loss  =  disparate_loss 

        self.log['dis_loss'] = disparate_loss.item()
        self.log['ins_loss'] = instance_loss.item()
        # self.log['cls_loss'] = class_loss.item()
        return loss, self.log
    
    def disparate_loss(self, z1,k2, y1,y2):
        k2 = concat_all_gather(k2)

        fz1,fz2 = self.filter(z1, k2, log=self.log)
        
        scale = 1/self.T
        logits = scale * self.filter.contrast(fz1,fz2)

        label = y1*self.num_classes+y2 # unique label for each pair
        pos_mask = (label.unsqueeze(1) == concat_all_gather(label).unsqueeze(0))
        c1_mask = (y1.unsqueeze(1) == concat_all_gather(y2).unsqueeze(0)) # exclude the key from class y1
        c2_mask = (y2.unsqueeze(1) == concat_all_gather(y2).unsqueeze(0)) # exclude the key from class y2
        class_mask = c1_mask|c2_mask

        loss = multipos_ce_loss(logits,pos_mask,class_mask)
        return loss
    

    def class_loss(self,z1,k2,y1,y2):
        k2 = concat_all_gather(k2)

        fz1,fz2 = self.filter(z1, k2, y1,log=self.log)

        scale = 1/self.T
        logits = scale * self.filter.contrast(fz1,fz2)

        pos_mask = (y1.unsqueeze(1) == concat_all_gather(y2).unsqueeze(0)) # exclude the key from class y1
        loss = multipos_ce_loss(logits,pos_mask,pos_mask)
        return loss

    

class MoCo_ResNet(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.fc.weight.shape[1]
        del self.base_encoder.fc, self.momentum_encoder.fc # remove original fc layer

        # projectors
        self.base_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim, False)


class MoCo_ViT(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.head.weight.shape[1]
        del self.base_encoder.head, self.momentum_encoder.head # remove original fc layer

        # projectors
        self.base_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim)


# utils
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
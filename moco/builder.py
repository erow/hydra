# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
from moco.filter import Filter, ConvFilter
import gin

@gin.configurable()
class PairSampler():
    def __init__(self,pair_file='res/semantic_distance.pth',max_level=11,alpha=0):
        """
        alpha=0: all class pairs are equally sampled
        alpha=10: almost identical class pairs
        """
        pair_samples = torch.load(pair_file)
        self.pair_samples = torch.tensor([x[0] for x in pair_samples if x[1]<max_level]) #n,2
        self.dist =  torch.tensor([x[1] for x in pair_samples if x[1]<max_level])
        self.weights = 1/(1+self.dist)**alpha
        self.weights = self.weights / self.weights.sum()
    
    def __call__(self, num_samples):
        sampled_indices = torch.multinomial(
                self.weights,
                num_samples,
                replacement=True
            )
        pairs = self.pair_samples[sampled_indices]
        dist = self.dist[sampled_indices]
        return pairs,dist
    

@gin.configurable(denylist=['dim','mlp_dim','T'])
class MoCo(nn.Module):
    """
    Build a MoCo model with a base encoder, a momentum encoder, and two MLPs
    https://arxiv.org/abs/1911.05722
        self.momentum = 0.9
    """
    def __init__(self, base_encoder, 
                 dim=256, mlp_dim=4096, T=1.0, 
                 alpha=0, beta=0.0, 
                 gamma=0.0,
                 norm='ln-none',
                 num_layers = 3,
                 warmup = 0,
                 sep = False,
                 grad_checkpointing=False,
                 num_classes=1000):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        """
        super(MoCo, self).__init__()

        self.T = T
        self.alpha=alpha
        self.beta = beta
        self.gamma = gamma
        self.sep = sep
        self.num_classes=num_classes
        self.num_layers = num_layers 
        
        # build encoders
        self.base_encoder = base_encoder(num_classes=mlp_dim)
        if grad_checkpointing:
            self.base_encoder.set_grad_checkpointing(True)
        
        self.momentum_encoder = base_encoder(num_classes=mlp_dim)
        self.filter = Filter(self.num_classes,dim)
        self.norm = norm
        self.warmup = warmup

        self._build_projector_and_predictor_mlps(dim, mlp_dim)
        if self.sep:
            self.predictor_dis = self._build_mlp(2, dim, mlp_dim, dim, norm.split('-')[1])
        self.scale_logit = nn.Parameter(torch.zeros(1)+np.log(20))
        
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient
            
        
        self.pair_sampler = PairSampler()
        
    
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
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        N = logits.shape[0]  # batch size per GPU
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        labels = (torch.arange(N, dtype=torch.long) + N * rank).to(q.device)
        return nn.CrossEntropyLoss()(logits, labels)

    def forward(self, images, m, targets,epoch):
        """
        Input:
            images: the list of image views
            m: moco momentum
        Output:
            loss
        """
        x1, x2 = images[0], images[1]

        self.log = {}
        
        
        # compute features
        xs = torch.cat([x1, x2], dim=0)
        z1,z2 = self.base_encoder(xs).chunk(2)
        q1 = self.predictor(z1)
        q2 = self.predictor(z2)
        with torch.no_grad():  # no gradient
            self._update_momentum_encoder(m)  # update the momentum encoder
            # compute momentum features as targets
            k1,k2 = self.momentum_encoder(xs).chunk(2)            

        instance_loss =  (self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1))/2
        loss  =  instance_loss
        
        # for local crops
        if len(images) > 2:
            local_loss = 0
            for i in range(2, len(images)):
                x = images[i]
                z = self.base_encoder(x)
                q = self.predictor(z)
                local_loss += (self.contrastive_loss(q, k1) + 
                               self.contrastive_loss(q, k2))/2
            loss += local_loss/ (len(images) - 2)
        

        ######### for disparate contrastive learning #########

        # pair generation
        # y1=[]
        # y2=[]
        # l=0
        # bs = len(q1)
        # while True:
        #     pairs, _ = self.pair_sampler(bs*2)
        #     y1_,y2_ = pairs.cuda().unbind(1)
        #     ## remove the samples not in the random pairs
        #     mask = (targets.unsqueeze(1) == y1_.unsqueeze(0)).any(0)
        #     mask &= (targets.unsqueeze(1) == y2_.unsqueeze(0)).any(0)
        #     y1.append(y1_[mask])
        #     y2.append(y2_[mask])
        #     l+=len(y1[-1])
        #     if l > bs:
        #         y1 = torch.cat(y1)[:bs]
        #         y2 = torch.cat(y2)[:bs]
        #         break
        
        #: random pair
        y1 = targets
        y2 = targets[torch.randperm(len(targets),device=targets.device)]

        
        if self.sep:
            # separated predictor
            q1 = self.predictor_dis(z1)
            q2 = self.predictor_dis(z2)

        if self.beta>0:
            # disparate contrast
            disparate_loss = (
                self.disparate_loss(q1,k2,y1,y2) + 
                self.disparate_loss(q2,k1,y1,y2))/2
            
            loss += self.beta * disparate_loss

        loss /= (1 + self.beta)

        C = np.log(len(k1)*( torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1))
        with torch.no_grad():
            activation, entropy = self.filter.gate.statistics()
            self.log['activation'] = activation.item()
            self.log['entropy'] = entropy.item()
            self.log['ins'] = C - instance_loss.item() 
            if self.beta>0:
                self.log['dis'] = C - disparate_loss.item() 
            self.log['z@sim'] = nn.functional.cosine_similarity(z1,z2).mean().item()
        
        if self.gamma>0:
            loss += self.gamma * entropy

        return loss, self.log
    
    def disparate_loss(self, z1, k2, y1, posy):
        k2 = concat_all_gather(k2)
        scale = 1/self.T
        # Filter.gated_contrast → moco.fused_gated_contrast (Triton/torch fused path)
        logits = scale * self.filter.gated_contrast(z1, k2, y1, posy)

        c1_mask = (y1.unsqueeze(1) == concat_all_gather(y1).unsqueeze(0)) # exclude samples from y1
        c2_mask = (posy.unsqueeze(1) == concat_all_gather(y1).unsqueeze(0)) # exclude samples from y2
        class_mask = c1_mask|c2_mask

        loss = multipos_ce_loss(logits,c2_mask,class_mask)
        return loss
    

    def class_loss(self,z1,k2,y1,y2):
        k2 = concat_all_gather(k2)
        scale = 1/self.T
        logits = scale * self.filter.gated_contrast(z1, k2, y1)

        pos_mask = (y1.unsqueeze(1) == concat_all_gather(y2).unsqueeze(0)) # exclude the key from class y1
        loss = multipos_ce_loss(logits,pos_mask,pos_mask)
        return loss

    

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



@torch.jit.script
def mean_representation(z, y, num_classes: int=25):
    # 105 µs ± 3.82 µs
    # Create one-hot encoding for the class labels
    one_hot = torch.nn.functional.one_hot(y, num_classes).float()
    
    # Calculate the sum of representations for each class
    class_sums = torch.matmul(one_hot.t(), z)
    
    # Calculate the count of each class
    class_counts = one_hot.sum(dim=0).unsqueeze(1)
    
    # Avoid division by zero by replacing zero counts with ones
    class_counts[class_counts == 0] = 1
    
    # Calculate the mean representation for each class
    class_vector = class_sums / class_counts
    
    return class_vector

def cross_cosine_similarity(x, y):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    sim = x @ y.t()
    return sim.mean()


if __name__ == "__main__":
    from torchvision.models import resnet18
    model = MoCo_ResNet(resnet18,sep=True,beta=1,grad_checkpointing=True)
    x1 = torch.randn(10,3,224,224)
    x2 = torch.randn(10,3,224,224)
    print( model(x1,x2,0.9,torch.randint(0,10,(10,)),0))
import math
import torch
from torch import nn
import torch.nn.functional as F
import gin


@gin.configurable()
class OpenGate(nn.Module):
    def __init__(self, embed_dim,num_classes):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes

    def forward(self,y1,y2=None,log=None):
        bs = y1.size(0)
        gate = torch.ones(bs,self.embed_dim,device=y1.device)
        return gate

@gin.configurable()
class BasicGate(OpenGate):
    def __init__(self, embed_dim, num_classes=1000, in_dim=512, mlp_dim=1024,
                 lam = 0, fuse=True):
        super().__init__(embed_dim,num_classes)
        
        self.mlp = nn.Sequential(
            nn.ReLU(), nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, mlp_dim),
            nn.ReLU(), nn.BatchNorm1d(mlp_dim),
            nn.Linear(mlp_dim, embed_dim),            
        )
        self.label_embedding = nn.Embedding(num_classes,in_dim)
        self.lam = lam
        self.fuse = fuse

    def statistics(self):
        labels = torch.arange(self.num_classes).cuda()
        label_embeds = self.label_embedding(labels)
        logits = self.mlp(label_embeds)
        gates = logits.sigmoid()
        activation = gates.sum(1).mean()
        entropy = torch.distributions.Bernoulli(gates).entropy().mean()
        return activation, entropy
    
    def forward(self,y1,y2=None,log=None):
        if self.fuse:      
            if y2 is None:
                label_embeds = self.label_embedding(y1)
            else:
                label_embeds = (self.label_embedding(y1) + self.label_embedding(y2))/2
            
            logits = self.mlp(label_embeds)
            gate = logits.sigmoid()
        else:
            if y2 is None:
                gate = self.mlp(self.label_embedding(y1)).sigmoid()
            else:
                gate1 = self.mlp(self.label_embedding(y1)).sigmoid()
                gate2 = self.mlp(self.label_embedding(y2)).sigmoid()
                gate = gate1 * gate2
        return gate


def reparameterize_with_gumbel_softmax(logits, tau=1.0, hard=False):
    """
    Apply the Gumbel-Softmax trick to reparameterize the Bernoulli distribution.
    logits: Logits from which to sample (e.g., the output of a neural network).
    tau: Temperature parameter. Lower values make samples more discrete.
    hard: If True, use straight-through Gumbel-Softmax Estimator.
    """
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
    y_soft = torch.sigmoid((logits + gumbel_noise) / tau)

    if hard:
        y_hard = torch.round(y_soft)
        y = y_hard - y_soft.detach() + y_soft
    else:
        y = y_soft

    return y

@gin.configurable
class GumbelGates(BasicGate):
    def forward(self, y1,y2=None,log=None):
        if y2 is None:
            label_embeds = self.label_embedding(y1)
        else:
            label_embeds = (self.label_embedding(y1) + self.label_embedding(y2))/2
        logits = self.mlp(label_embeds)
        gate = reparameterize_with_gumbel_softmax(logits)

        if not log is None:
            p = logits.sigmoid().detach().clamp(1e-6,1-1e-6)
            dist = torch.distributions.Bernoulli(p)
            entropy = dist.entropy()
            log['entropy'] = entropy.mean().item()
            open = (gate>0.5).float()
            log['activation']=(open.sum(0)>20).float().sum().item()
            
        return gate
    
@gin.configurable()
class StochasticGate(BasicGate):
    """Paper: Feature Selection using Stochastic Gates
    Code: https://github.com/runopti/stg/blob/master/python/stg/models.py
    """
    def __init__(self, embed_dim, num_classes=1000, in_dim=512, mlp_dim=1024,
                sigma=0.5,lam=1e-6):
        super().__init__(embed_dim, num_classes,in_dim,mlp_dim)
        self.sigma=sigma
        self.lam = lam
        
    def forward(self,y1,y2=None,log=None):
        if y2 is None:
            label_embeds = self.label_embedding(y1)
        else:
            label_embeds = (self.label_embedding(y1) + self.label_embedding(y2))/2
        mu = self.mlp(label_embeds)
        noise = torch.randn_like(mu)*self.sigma
        gate = self.hard_sigmoid(mu+noise)
        if not log is None:
            reg = self.regularizer((mu+0.5)/self.sigma).sum(1).mean()
            log['reg'] = reg
            log['mu']=mu.mean().item()+0.5
            # log['lam']=self.lam
        return gate


    def hard_sigmoid(self, x):
        return torch.clamp(x+0.5, 0.0, 1.0)

    def regularizer(self, x):
        ''' Gaussian CDF. '''
        return 0.5 * (1 + torch.erf(x / math.sqrt(2))) 
    
    def reg(self,mu):
        reg = torch.mean(self.regularizer((mu + 0.5)/self.sigma)) 
        return reg


@gin.configurable()
class Filter(nn.Module):
    def __init__(self,num_classes, embed_dim, gate_fn=BasicGate):
        super().__init__()
        self.embed_dim = embed_dim
        self.gate = gate_fn(embed_dim,num_classes=num_classes)
    
    def forward(self, x1,x2,y1,y2=None):
        gate = self.gate(y1,y2)
        x1 = torch.einsum("bk,bk->bk",x1,gate)
        x2 = torch.einsum("nk,bk->bnk",x2,gate)
        x1 =  F.normalize(x1,p=2,dim=-1)
        x2 =  F.normalize(x2,p=2,dim=-1)
        return x1, x2
    
    def contrast(self,x1,x2):
        logits =  torch.einsum("bj,bnj->bn",x1,x2)
        return logits

    def gated_contrast(self, x1, x2, y1, y2=None, *, backend: str = "auto"):
        """Gate → normalize → contrast in one shot (optionally Triton-fused).

        Equivalent to ``contrast(*forward(x1, x2, y1, y2))`` but avoids
        materializing the ``[B, N, K]`` gated-key tensor when ``backend`` is
        ``"auto"`` / ``"triton"`` / ``"torch"``.
        """
        from moco.fused_gated_contrast import fused_gated_contrast

        gate = self.gate(y1, y2)
        return fused_gated_contrast(x1, x2, gate, backend=backend)

from timm.models.convnext  import convnextv2_atto
class VisionGate(nn.Module):
    def __init__(self, embed_dim, in_dim=512, mlp_dim=1024,
                 lam = 0, fuse=True):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.ReLU(), nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, mlp_dim),
            nn.ReLU(), nn.BatchNorm1d(mlp_dim),
            nn.Linear(mlp_dim, embed_dim),            
        )
        self.label_embedding = convnextv2_atto(num_classes=embed_dim)
        self.lam = lam
        self.fuse = fuse
    
    def forward(self,y1,y2=None,log=None):
        if y2 is None:
            label_embeds = self.label_embedding(y1)
        else:
            label_embeds = (self.label_embedding(y1) + self.label_embedding(y2))/2
        
        logits = self.mlp(label_embeds)
        gate = logits.sigmoid()

        if not log is None:
            p = gate.detach()
            dist = torch.distributions.Bernoulli(p)
            entropy = dist.entropy()
            log['entropy'] = entropy.mean().item()
            open = (gate>0.5).float()
            log['activation']=(open.sum(0)>20).float().sum().item()
            
        return gate
    
class ConvFilter(nn.Module):
    def __init__(self,num_classes, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.gate = VisionGate(embed_dim)
        

    def forward(self, x1,x2, log=None):
        gate = self.gate(x1,x2,log=log)
        x1 = torch.einsum("bk,bk->bk",x1,gate)
        x2 = torch.einsum("nk,bk->bnk",x2,gate)
        x1 =  F.normalize(x1,p=2,dim=-1)
        x2 =  F.normalize(x2,p=2,dim=-1)
        return x1, x2
    
    def contrast(self,x1,x2):
        logits =  torch.einsum("bj,bnj->bn",x1,x2)
        return logits
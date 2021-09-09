import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class BinMultitask(nn.Module):
    def __init__(self, backbone, feature_size, threshold1, threshold2, ordinal = True):
        super(BinMultitask, self).__init__()
        self.backbone = backbone
        self.W = nn.Parameter(torch.empty(feature_size, 1))
        nn.init.normal_(self.W)
        self.thr1 = threshold1
        self.thr2 = threshold2
        self.softmax = nn.Softmax(dim = 1)
        self.max = 2*self.thr2
        
        if ordinal == True:
            self.min = -self.thr2
        else:
            self.min = 0
        
    def forward(self, x):
        # Feature Extractor
        embed = self.backbone(x)        
        
        # Feature Extractor
        out = torch.matmul(embed, self.W)
        score = torch.clamp(out, min=self.min, max = self.max)                      
        dist1 = self.thr1 - score
        dist2 = self.thr2 - score
        
        # Logit Extractor for Ordinal Regression (urban / rural / uninhabited)
        logit = torch.cat((-dist2, torch.min(-dist1, dist2), dist1), dim = 1) 
        return embed, score, self.softmax(logit)
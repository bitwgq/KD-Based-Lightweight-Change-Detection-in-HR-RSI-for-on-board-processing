import torch.nn as nn
import torch
from torch.nn import functional as F


class CriterionPC(nn.Module):
    def __init__(self, classes):
        super(CriterionPC, self).__init__()
        self.num_classes = classes

    def forward(self, preds_S, preds_T, target):
        feat_S = preds_S
        feat_T = preds_T
        feat_S = F.normalize(feat_S.view(-1, 16, 256*256), dim=1).view(-1, 16, 256, 256)
        feat_T = F.normalize(feat_T.view(-1, 16, 256*256), dim=1).view(-1, 16, 256, 256)
        feat_T.detach()
        size_f = (feat_S.shape[2], feat_S.shape[3])
        tar_feat_S = nn.Upsample(size_f, mode='nearest')(target.unsqueeze(1).float()).expand(feat_S.size())
        tar_feat_T = nn.Upsample(size_f, mode='nearest')(target.unsqueeze(1).float()).expand(feat_T.size())
        center_feat_S = feat_S.clone()
        center_feat_T = feat_T.clone()
        center_feat_S_FS = feat_S.clone()
        center_feat_T_FS = feat_T.clone()
        for i in range(self.num_classes):
            mask_feat_S = (tar_feat_S == i).float()
            mask_feat_T = (tar_feat_T == i).float()
            center_feat_S = (1 - mask_feat_S) * center_feat_S + mask_feat_S * ((mask_feat_S * feat_S).sum(-1).sum(-1) / (mask_feat_S.sum(-1).sum(-1) + 1e-6)).unsqueeze(-1).unsqueeze(-1)
            center_feat_T = (1 - mask_feat_T) * center_feat_T + mask_feat_T * ((mask_feat_T * feat_T).sum(-1).sum(-1) / (mask_feat_T.sum(-1).sum(-1) + 1e-6)).unsqueeze(-1).unsqueeze(-1)
            center_feat_S_FS = mask_feat_S * center_feat_S_FS + (1 - mask_feat_S) * ((mask_feat_S * feat_S).sum(-1).sum(-1) / (mask_feat_S.sum(-1).sum(-1) + 1e-6)).unsqueeze(-1).unsqueeze(-1)
            center_feat_T_FS = mask_feat_T * center_feat_T_FS + (1 - mask_feat_T) * ((mask_feat_T * feat_T).sum(-1).sum(-1) / (mask_feat_T.sum(-1).sum(-1) + 1e-6)).unsqueeze(-1).unsqueeze(-1)

        cos = nn.CosineSimilarity(dim=1)
        center_feat_S.detach()
        center_feat_S_FS.detach()
        pcsim_feat_S = cos(feat_S, center_feat_S) - cos(feat_S, center_feat_S_FS)
        pcsim_feat_T = cos(feat_T, center_feat_T) - cos(feat_T, center_feat_T_FS)
        pcsim_feat_S = torch.exp(pcsim_feat_S)
        pcsim_feat_T = torch.exp(pcsim_feat_T)
        pcsim_feat_T.detach()
        mse = nn.MSELoss()
        loss = mse(pcsim_feat_S, pcsim_feat_T)
        return loss

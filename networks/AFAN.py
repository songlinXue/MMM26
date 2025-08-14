import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def calc_ins_mean_std(x, eps=1e-5):
    size = x.size()
    assert len(size) == 4
    N, C = size[:2]

    if x.numel() == 0 or x.reshape(N, C, -1).size(-1) <= 1:
        mean = torch.zeros(N, C, 1, 1, device=x.device, dtype=x.dtype)
        std  = torch.ones (N, C, 1, 1, device=x.device, dtype=x.dtype)
        return mean, std
        
    # reshape
    var = x.reshape(N, C, -1).var(dim=2) + eps
    std = var.sqrt().reshape(N, C, 1, 1)
    mean = x.reshape(N, C, -1).mean(dim=2).reshape(N, C, 1, 1)
    return mean, std

def instance_norm_mix(content_feat, style_feat):

    if content_feat.size()[2:] != style_feat.size()[2:]:
        style_feat = F.interpolate(style_feat, size=content_feat.shape[2:], mode='nearest')
    
    style_mean, style_std = calc_ins_mean_std(style_feat)
    content_mean, content_std = calc_ins_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean) / content_std
    return normalized_feat * style_std + style_mean

def cn_rand_bbox(size, beta, bbx_thres, is_fraud_detection=True, max_try=100):
    """
    size: (B, C, H, W) 或 (H, W)
    beta: Beta
    """
    if len(size) == 4:
        _, _, H, W = size
    else:
        H, W = size[-2], size[-1]

    edge_ratio = 0.1 if is_fraud_detection else 0.0
    for _ in range(max_try):
        cut_rat = np.sqrt(np.random.beta(beta, beta)) * (edge_ratio + (1 - edge_ratio) * 0.5)
        cut_w = max(1, int(W * cut_rat))
        cut_h = max(1, int(H * cut_rat))

        if is_fraud_detection:
            side = np.random.choice(['top', 'bottom', 'left', 'right'])
            if side == 'top':
                cy = np.random.randint(0, cut_h // 2)
                cx = np.random.randint(0, W)
            elif side == 'bottom':
                cy = np.random.randint(H - cut_h // 2, H)
                cx = np.random.randint(0, W)
            elif side == 'left':
                cx = np.random.randint(0, cut_w // 2)
                cy = np.random.randint(0, H)
            else:  # right
                cx = np.random.randint(W - cut_w // 2, W)
                cy = np.random.randint(0, H)
        else:
            cx = np.random.randint(W)
            cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        ratio = (bbx2 - bbx1) * (bby2 - bby1) / (W * H)
        if ratio >= bbx_thres:
            return (bbx1, bby1, bbx2, bby2)

    return (0, 0, W, H)

class AttentionGuidedROI(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 16, in_channels, 1),
            nn.Sigmoid()
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )
        self.min_roi_ratio = 0.05

    def get_bbox_from_mask(self, mask):
        B = mask.size(0)
        rois = []
        for i in range(B):
            non_zero = (mask[i] > 0).nonzero()
            if non_zero.numel() == 0:
                B, H, W = mask.size()
                rois.append((0, 0, W, H))
                continue
                
            y_min = non_zero[:, 0].min().item()
            y_max = non_zero[:, 0].max().item()
            x_min = non_zero[:, 1].min().item()
            x_max = non_zero[:, 1].max().item()
            
            B, H, W = mask.size()
            min_size = min(H, W) * self.min_roi_ratio
            if (x_max - x_min) < min_size or (y_max - y_min) < min_size:
                size = int(min_size)
                cx = (x_min + x_max) // 2
                cy = (y_min + y_max) // 2
                x_min = max(0, cx - size//2)
                x_max = min(W, cx + size//2)
                y_min = max(0, cy - size//2)
                y_max = min(H, cy + size//2)
                
            rois.append((int(x_min), int(y_min), int(x_max), int(y_max)))
        return rois

    def forward(self, x):
        channel_attention = self.se(x)  # [B, C, 1, 1]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        spatial_attention = self.spatial(spatial_input)
        attention = channel_attention * spatial_attention
        mean_attention = torch.mean(attention, dim=1, keepdim=True)
        mean_mask = (mean_attention > 0.5).squeeze(1)
        rois = self.get_bbox_from_mask(mean_mask)
        return rois, attention

class CrossNorm(nn.Module):
    def __init__(self, crop=None, beta=1):
        super().__init__()
        self.crop = crop
        self.beta = beta

    def forward(self, x, rois=None):
        if not self.training or self.crop is None:
            return x

        B, C, H, W = x.size()
        ins_idxs = torch.randperm(B).to(x.device)
        output = x.clone()

        for i in range(B):
            if self.crop in ['content', 'both']:
                if rois is not None:
                    x1, y1, x2, y2 = rois[i]
                else:
                    x1, y1, x2, y2 = cn_rand_bbox(
                        x.size(), self.beta, 0.1,
                        is_fraud_detection=(self.crop == 'both')
                    )
                content_region = x[i, :, y1:y2, x1:x2]
            else:
                content_region = x[i]
        
            style_idx = ins_idxs[i]
            if self.crop in ['style', 'both']:
                sx1, sy1, sx2, sy2 = cn_rand_bbox(
                    x.size(), self.beta, 0.1,
                    is_fraud_detection=False
                )
                style_region = x[style_idx, :, sy1:sy2, sx1:sx2]
            else:
                style_region = x[style_idx]
        
            if content_region.numel() < 4 or style_region.numel() < 4:
                continue
        
            mixed_region = instance_norm_mix(
                content_region.unsqueeze(0),
                style_region.unsqueeze(0)
            )
        
            if self.crop in ['content', 'both']:
                if rois is not None:
                    x1, y1, x2, y2 = rois[i]
                    output[i, :, y1:y2, x1:x2] = mixed_region.squeeze(0)
                else:
                    output[i, :, y1:y2, x1:x2] = mixed_region.squeeze(0)
            else:
                output[i] = mixed_region.squeeze(0)
                
        return output

class SelfNorm(nn.Module):
    def __init__(self, chan_num, is_two=False, use_roi=False):
        super().__init__()
        self.g_fc = nn.Conv1d(chan_num, chan_num, 2, groups=chan_num, bias=False)
        self.g_bn = nn.BatchNorm1d(chan_num)
        self.f_fc = nn.Conv1d(chan_num, chan_num, 2, groups=chan_num, bias=False) if is_two else None
        self.f_bn = nn.BatchNorm1d(chan_num) if is_two else None
        self.use_roi = use_roi

    def forward(self, x, rois=None, attention_map=None):
        B, C, H, W = x.size()
        
        if self.use_roi and rois is not None and attention_map is not None:
            # 使用ROI区域计算统计特征
            stats_list = []
            for i in range(B):
                x1, y1, x2, y2 = rois[i]
                roi_feat = x[i:i+1, :, y1:y2, x1:x2]
                
                # 使用注意力图加权
                roi_attention = attention_map[i:i+1, :, y1:y2, x1:x2]
                roi_attention = roi_attention / (roi_attention.sum() + 1e-8)
                
                # 计算加权均值和标准差
                mean = (roi_feat * roi_attention).sum(dim=[2, 3], keepdim=True)
                std = torch.sqrt(
                    ((roi_feat - mean)**2 * roi_attention).sum(dim=[2, 3], keepdim=True) + 1e-5)
                
                stats_list.append(torch.cat([mean, std], dim=1))
            
            stats = torch.cat(stats_list, dim=0)
            mean = stats[:, :C, :, :]
            std = stats[:, C:, :, :]
        else:
            mean = torch.mean(x, dim=[2, 3], keepdim=True)
            std = torch.std(x, dim=[2, 3], keepdim=True)
        
        mean_vec = mean.squeeze(-1).squeeze(-1)  # [B, C]
        std_vec = std.squeeze(-1).squeeze(-1)    # [B, C]
        statistics = torch.cat([mean_vec, std_vec], dim=1)  # [B, 2*C]
        
        statistics = statistics.view(B, C, 2)
        
        g_y = self.g_fc(statistics)  # [B, C, 1]
        g_y = self.g_bn(g_y)
        g_y = torch.sigmoid(g_y).view(B, C, 1, 1)  # [B, C, 1, 1]
        
        if self.f_fc is not None:
            f_y = self.f_fc(statistics)  # [B, C, 1]
            f_y = self.f_bn(f_y)
            f_y = torch.sigmoid(f_y).view(B, C, 1, 1)  # [B, C, 1, 1]
            
            return x * g_y + mean * (f_y - g_y)
        else:
            return x * g_y

class AFAN(nn.Module):
    def __init__(self, crossnorm_params=None, selfnorm_params=None, use_roi_in_selfnorm=True):
        super().__init__()
        self.crossnorm = CrossNorm(**(crossnorm_params or {'crop': 'both'}))
        self.attention_roi = AttentionGuidedROI(in_channels=512)
        
        selfnorm_params = selfnorm_params or {}
        selfnorm_params['use_roi'] = use_roi_in_selfnorm
        self.selfnorm = SelfNorm(chan_num=512, **selfnorm_params)
        
        self.use_roi_in_selfnorm = use_roi_in_selfnorm

    def forward(self, x):
        rois, attention_map = self.attention_roi(x)
        
        x = self.crossnorm(x, rois=rois)
        
        if self.use_roi_in_selfnorm:
            x = self.selfnorm(x, rois=rois, attention_map=attention_map)
        else:
            x = self.selfnorm(x)
            
        return x

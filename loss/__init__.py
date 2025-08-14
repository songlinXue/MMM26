import torch
import torch.nn as nn
import torch.nn.functional as F

class BDCLoss(nn.Module):
    def __init__(self, num_classes, feature_dim, alpha=1.0, lambda_adv=0.5, margin=0.5, device='cuda'):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.alpha = alpha
        self.lambda_adv = lambda_adv
        self.margin = margin
        self.centers = nn.Parameter(torch.randn(num_classes, feature_dim).to(device))
    
    def forward(self, features, labels):
        batch_size = features.size(0)
        centers_batch = self.centers[labels.long()]

        similarity = F.cosine_similarity(features, centers_batch, dim=1)  # [B]
        exp_decay = torch.exp(self.alpha * similarity)
        intra_loss = F.mse_loss(features, centers_batch, reduction='none').sum(dim=1) / exp_decay
        intra_loss = intra_loss.mean()

        sim_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2)  # [B,B]
        mask = labels.unsqueeze(1) != labels.unsqueeze(0)
        sim_adv = sim_matrix[mask]  # [N*(N-1)/2]
        adv_loss = F.relu(self.margin - sim_adv).mean()

        loss = intra_loss + self.lambda_adv * adv_loss
        return loss
class DynamicSoftKMeansLoss(nn.Module):
    def __init__(self, feature_dim, max_centers=5, margin=0.5, temperature=1.0, device=None):
        super().__init__()
        self.feature_dim = feature_dim
        self.max_centers = max_centers
        self.margin = margin
        self.temperature = temperature
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.centers = nn.Parameter(torch.randn(max_centers, feature_dim).to(self.device))

    def forward(self, feat_normed, labels, label2):
        """
        Args:
            feat_normed (Tensor): shape [B, D]
            labels (Tensor): shape [B]
            label2 (Tensor): shape [B]
        """
        unique_labels = torch.unique(labels)
        total_loss = 0.0
        
        for c in unique_labels:
            mask = (labels == c) & (label2 == 1)
            if mask.sum() == 0:
                continue

            class_feats = feat_normed[mask]  # [N_c, D]

            distances = torch.cdist(class_feats, self.centers)

            probs = F.softmax(-distances / self.temperature, dim=1)  # [N_c, K]

            weighted_dist = (probs * distances).sum(dim=1)  # [N_c]
            compactness_loss = (weighted_dist ** 2).mean()

            closest_center_idx = torch.argmin(distances.mean(dim=0))
            
            other_center_mask = torch.ones(self.max_centers, dtype=torch.bool, device=self.device)
            other_center_mask[closest_center_idx] = False
            other_centers = self.centers[other_center_mask]  # [K-1, D]

            other_dists = torch.cdist(class_feats.unsqueeze(1), other_centers.unsqueeze(0))
            
            min_other_dist = other_dists.min(dim=2).values.squeeze()  # [N_c]

            violations = F.relu(weighted_dist + self.margin - min_other_dist)
            separation_loss = violations.mean()

            total_loss += compactness_loss + separation_loss

        return total_loss / len(unique_labels) if len(unique_labels) > 0 else torch.tensor(0.0, device=self.device)


class TripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()


class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, features, labels):
        batch_size = features.size(0)
        distmat = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(features, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss

class ContrastLoss(nn.Module):
    
    def __init__(self):
        super(ContrastLoss, self).__init__()
        pass

    def forward(self, anchor_fea, reassembly_fea, contrast_label):
        contrast_label = contrast_label.float()
        anchor_fea = anchor_fea.detach()
        loss = -(F.cosine_similarity(anchor_fea, reassembly_fea, dim=-1))
        loss = loss*contrast_label
        return loss.mean()


def AAMS(logits, spoof_label, type_label, num_classes):
    s = 30
    log_terms = []
    type_label = type_label * 2
    spoof_label = list(spoof_label.data.cpu().numpy()) * 2
    for i, logit in enumerate(logits):
        cls = type_label[i]
        if spoof_label[i] == 1:# live
            m = 0.4
        else:# spoof
            m = 0.1
        pos_mask = F.one_hot(torch.Tensor([cls]).long(), num_classes=num_classes)[0].to(logits.device)
        neg_mask = 1 - pos_mask

        logit_am = s * (logit - pos_mask * m)
        logit_max = torch.max(logit_am)
        logit_am = logit_am - logit_max.detach()

        pos_term = (logit_am * pos_mask).sum() / (pos_mask.sum() + 1e-10)
        neg_term = (torch.exp(logit_am)).sum()

        log_term = pos_term - torch.log(neg_term + 1e-15)
        log_terms.append(log_term)

    loss = -sum(log_terms) / len(log_terms)
    return loss

def feat_sim_loss(feat1, feat2):
    return torch.norm(feat1 - feat2, dim=1).mean()


def supcon_loss(features, labels=None, mask=None, temperature = 0.1):
    base_temperature = 0.07
    """Compute loss for model. If both `labels` and `mask` are None,
    it degenerates to SimCLR unsupervised loss:
    https://arxiv.org/pdf/2002.05709.pdf
    Args:
        features: hidden vector of shape [bsz, n_views, ...].
        labels: ground truth of shape [bsz].
        mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
            has the same class as sample i. Can be asymmetric.
    Returns:
        A loss scalar.
    """
    device = (torch.device('cuda')
              if features.is_cuda
              else torch.device('cpu'))

    if len(features.shape) < 3:
        raise ValueError('`features` needs to be [bsz, n_views, ...],'
                         'at least 3 dimensions are required')
    if len(features.shape) > 3:
        features = features.view(features.shape[0], features.shape[1], -1)

    batch_size = features.shape[0]
    if labels is not None and mask is not None:
        raise ValueError('Cannot define both `labels` and `mask`')
    elif labels is None and mask is None:
        mask = torch.eye(batch_size, dtype=torch.float32).to(device)
    elif labels is not None:
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)
    else:
        mask = mask.float().to(device)

    contrast_count = features.shape[1]
    contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
    anchor_feature = contrast_feature
    anchor_count = contrast_count

    # compute logits
    anchor_dot_contrast = torch.div(
        torch.matmul(anchor_feature, contrast_feature.T),
    temperature)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # tile mask
    mask = mask.repeat(anchor_count, contrast_count)
    # mask-out self-contrast cases
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
        0
    )
    mask = mask * logits_mask

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

    # loss
    loss = - (temperature / base_temperature) * mean_log_prob_pos
    loss = loss.view(anchor_count, batch_size).mean()

    return loss



def simclr_loss(features):

    labels = torch.cat([torch.arange(len(features) // 2) for i in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    device = (torch.device('cuda')
              if features.is_cuda
              else torch.device('cpu'))
    labels = labels.to(device)

    # features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / 0.7
    criterion = torch.nn.CrossEntropyLoss().to(device)
    return criterion(logits, labels)

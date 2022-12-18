import torch


class KpLoss(object):
    def __init__(self):
        self.criterion = torch.nn.MSELoss(reduction='none')

    def __call__(self, logits, targets):
        assert len(logits.shape) == 4, 'logits should be 4-ndim'
        device = logits.device
        bs = logits.shape[0] # 4-dim的logits的第一维度是batch_size。(bs,num_kps,h,w)
        # [num_kps, H, W] -> [B, num_kps, H, W],t:target，取出归一化后的heatmap堆叠(stack)成一个tensor，每一个target里有num_kps个heatmap。
        heatmaps = torch.stack([t["heatmap"].to(device) for t in targets])
        # [num_kps] -> [B, num_kps]
        kps_weights = torch.stack([t["kps_weights"].to(device) for t in targets])

        # [B, num_kps, H, W] -> [B, num_kps]
        loss = self.criterion(logits, heatmaps).mean(dim=[2, 3]) # 先从dim=3(w,逐层按行)求均值，然后再按照dim=2（h:按列求句均值），这样就是整个hxw的均值
        loss = torch.sum(loss * kps_weights) / bs
        return loss

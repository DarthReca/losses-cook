import torch
from torch import einsum, nn


class GDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, smooth=1e-5, ignore_index=-100):
        """
        Generalized Dice;
        Copy from: https://github.com/LIVIAETS/surface-loss/blob/108bd9892adca476e6cdf424124bc6268707498e/losses.py#L29
        paper: https://arxiv.org/pdf/1707.03237.pdf
        tf code: https://github.com/NifTK/NiftyNet/blob/dev/niftynet/layer/loss_segmentation.py#L279
        """
        super(GDiceLoss, self).__init__()

        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, net_output, gt):
        # Adaptation for 1 channel
        net_output = net_output.permute(0, 2, 1)
        gt = gt.unsqueeze(1)

        net_output = net_output.reshape(*net_output.shape, 1, 1)
        gt = gt.reshape(*gt.shape, 1, 1)

        shp_x = net_output.shape  # (batch size,class_num,x,y,z)
        shp_y = gt.shape  # (batch size,1,x,y,z)
        # Ignore index
        mask = gt != self.ignore_index
        net_output = net_output * mask
        gt = gt * mask
        # one hot code for gt
        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                gt = gt.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = gt
            else:
                gt = gt.long()
                y_onehot = torch.zeros(shp_x)
                if net_output.device.type == "cuda":
                    y_onehot = y_onehot.cuda(net_output.device.index)
                y_onehot.scatter_(1, gt, 1)

        if self.apply_nonlin is not None:
            net_output = self.apply_nonlin(net_output)

        # copy from https://github.com/LIVIAETS/surface-loss/blob/108bd9892adca476e6cdf424124bc6268707498e/losses.py#L29
        w: torch.Tensor = (
            1 / (einsum("bcxyz->bc", y_onehot).type(torch.float32) + 1e-10) ** 2
        )
        intersection: torch.Tensor = w * einsum(
            "bcxyz, bcxyz->bc", net_output, y_onehot
        )
        union: torch.Tensor = w * (
            einsum("bcxyz->bc", net_output) + einsum("bcxyz->bc", y_onehot)
        )
        divided: torch.Tensor = (
            -2
            * (einsum("bc->b", intersection) + self.smooth)
            / (einsum("bc->b", union) + self.smooth)
        )
        gdc = divided.mean()

        return gdc
class MultiLabelDiceLoss(nn.Module):
    """
    Dice loss designed for multi-label classification.
    It compares a vector of predicted logits with a binary ground truth vector.
    """
    def __init__(self, smooth=1e-5):
        super(MultiLabelDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, true_labels):
        # Apply sigmoid to the logits to get independent probabilities (0-1) for each class
        probs = torch.sigmoid(logits)
        
        # Ensure the true labels are floats for the calculation
        true_labels = true_labels.float()
        
        # Calculate the intersection and the total number of "present" labels
        intersection = (probs * true_labels).sum()
        total_present = probs.sum() + true_labels.sum()
        
        # Calculate the Dice coefficient
        dice_coeff = (2. * intersection + self.smooth) / (total_present + self.smooth)
        
        # The loss is 1 - the Dice coefficient
        return 1 - dice_coeff
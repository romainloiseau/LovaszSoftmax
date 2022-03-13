import torch
import torch.nn.functional as F

class LovaszSoftmaxFast(torch.jit.ScriptModule):

    def __init__(self, n_classes=20, ignore_index=None):
        super().__init__()

        if ignore_index is not None:
            self.register_buffer("ignore_index", torch.tensor(ignore_index))
        else:
            self.ignore_index = None
        self.register_buffer("class_to_sum", torch.arange(n_classes, dtype=torch.int64))
        
    @torch.jit.script_method
    def forward(self, probas: torch.Tensor, labels: torch.Tensor):
        if probas.dim() == 3:
            # assumes output of a sigmoid layer
            probas = probas.unsqueeze(1)
        probas = probas.flatten(-2, -1).permute(0, 2, 1)
        labels = labels.flatten(-2, -1)
        
        if self.ignore_index is not None:
            valid = labels != self.ignore_index

            probas = probas[valid]
            labels = labels[valid]

        if probas.numel() == 0:
            return probas * 0.

        all_fg = (labels.unsqueeze(-1) == self.class_to_sum).float()
        all_gts = all_fg.sum(0)
        has_label = all_gts != 0

        all_fg = all_fg[:, has_label]
        probas = probas[:, has_label]
        all_gts = all_gts[has_label]       

        all_errors = (all_fg - probas).abs()

        all_errors_sorted, all_perm = torch.sort(all_errors, 0, descending=True)
        all_perm = all_perm.data

        all_fg_sorted = torch.gather(all_fg, 0, all_perm)
        
        all_intersection = all_gts - all_fg_sorted.cumsum(0)
        all_union = all_gts + (1. - all_fg_sorted).cumsum(0)

        all_jaccard = 1. - all_intersection / all_union
        all_jaccard[1:] = all_jaccard[1:] - all_jaccard[:-1]

        losses = (all_errors_sorted * all_jaccard).sum(0)

        return losses.mean()
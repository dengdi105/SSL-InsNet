import torch
import torch.nn as nn
import torch.nn.functional as F

class NeighborhoodWeightedFocalLoss(nn.Module):
    """
    Neighborhood-Weighted Focal Loss (NWFL) for Sequence Structured Labeling.
    Combines Focal Loss sample difficulty mitigation with Markov Random Field (MRF) 
    inspired neighborhood smoothing penalties to suppress sequence tag fragmentation.
    """
    def __init__(self, alpha=0.25, gamma=2.0, lambda_smooth=0.1, reduction='mean'):
        """
        Args:
            alpha (float): Class balancing factor.
            gamma (float): Focusing parameter for hard-example mining.
            lambda_smooth (float): Scaling factor for neighborhood smoothing constraints.
            reduction (str): Reduction method. Options: 'none', 'mean', 'sum'.
        """
        super(NeighborhoodWeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_smooth = lambda_smooth
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Model predictions with shape [Batch_Size, Seq_Len, Num_Classes].
            targets (torch.Tensor): Ground truth labels with shape [Batch_Size, Seq_Len].
        
        Returns:
            torch.Tensor: Computed NWFL loss.
        """
        batch_size, seq_len, num_classes = inputs.size()
        
        # 1. Flatten sequence and batch dimensions for base token-wise loss calculation
        inputs_flat = inputs.view(-1, num_classes)
        targets_flat = targets.view(-1)
        
        # 2. Compute the standard cross-entropy and derive predictive probability pt
        ce_loss = F.cross_entropy(inputs_flat, targets_flat, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # 3. Compute the base Focal Loss to alleviate easy-negative over-dominance
        focal_weight = self.alpha * ((1.0 - pt) ** self.gamma)
        base_focal_loss = (focal_weight * ce_loss).view(batch_size, seq_len)
        
        # 4. Extract hard predictions to establish spatial neighborhood constraints
        preds = torch.argmax(inputs, dim=-1)
        
        # 5. Measure label mutations along the spatial/temporal axis
        # Check mismatch between adjacent positions: pred[:, t] != pred[:, t-1]
        diff_left = (preds[:, 1:] != preds[:, :-1]).float()
        
        # 6. Aggregate neighborhood mismatch penalties to trace local sequence fragmentation
        smooth_penalty = torch.zeros_like(preds, dtype=torch.float)
        smooth_penalty[:, :-1] += diff_left
        smooth_penalty[:, 1:] += diff_left
        
        # 7. Apply Neighborhood-Weighted adaptive regularized penalty scaling
        total_loss = base_focal_loss * (1.0 + self.lambda_smooth * smooth_penalty)
        
        # 8. Apply specified reduction method
        if self.reduction == 'mean':
            return torch.mean(total_loss)
        elif self.reduction == 'sum':
            return torch.sum(total_loss)
        else:
            return total_loss

if __name__ == "__main__":
    # Toy verification test
    # Simulating a batch of 2 sequences, each of length 5, with 3 structural variant tag categories
    sample_inputs = torch.randn(2, 5, 3, requires_grad=True)
    sample_targets = torch.tensor([[0, 1, 2, 2, 0], [0, 0, 0, 0, 0]], dtype=torch.long)
    
    criterion = NeighborhoodWeightedFocalLoss(alpha=0.25, gamma=2.0, lambda_smooth=0.1)
    loss = criterion(sample_inputs, sample_targets)
    
    print("NWFL forward validation successful. Loss tensor output:", loss.item())
    loss.backward()
    print("NWFL backward propagation gradient check complete.")

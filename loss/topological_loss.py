import torch
import torch.nn as nn
import torch.nn.functional as F
import geomloss

class TopologicalLoss(nn.Module):
    """
    A differentiable loss function that compares the geometric structure of two
    point clouds using the Sinkhorn divergence (a Wasserstein approximation).
    """
    def __init__(self, p=2, blur=0.05):
        """
        Initializes the loss module.
        Args:
            p (int): The exponent for the ground cost (e.g., p=2 for squared Euclidean distance).
            blur (float): The regularization strength for the Sinkhorn algorithm.
        """
        super(TopologicalLoss, self).__init__()
        # Initialize the geomloss function once for efficiency.
        self.loss_fn = geomloss.SamplesLoss("sinkhorn", p=p, blur=blur, backend="tensorized")

    def _create_point_cloud(self, source_tensor, mask, embedding_matrix):
        """
        Creates a batch of point clouds from either logits (soft) or labels (hard).
        """
        if source_tensor.dim() == 3: # Logits
            # Create "soft" embeddings from logits
            probs = F.softmax(source_tensor, dim=-1)
            embeddings = torch.matmul(probs, embedding_matrix)
        else: # Labels
            # Create standard "hard" embeddings from labels
            labels_for_embedding = source_tensor.clone()
            labels_for_embedding[source_tensor == -100] = 0  # Replace -100 with a valid index
            embeddings = F.embedding(labels_for_embedding, embedding_matrix)
            embeddings[source_tensor == -100] = 0  # Zero out embeddings for ignored tokens
        
        # For each item in the batch, select the embeddings where the mask is 1
        point_clouds = [emb[mask[i]] for i, emb in enumerate(embeddings)]
        
        return point_clouds

    def forward(self, logits, labels, ingredients_mask, embedding_matrix):
        """
        Calculates the topological loss on the ingredients.
        
        Args:
            logits (Tensor): Raw logits from the model [batch_size, seq_len, vocab_size].
            labels (Tensor): Ground truth token IDs [batch_size, seq_len].
            ingredients_mask (Tensor): Binary mask [batch_size, seq_len] with 1s at ingredient positions.
            embedding_matrix (Tensor): The model's input embedding matrix [vocab_size, embedding_dim].
            
        Returns:
            Tensor: A scalar loss value.
        """
        # Ensure mask sequence length matches logits sequence length
        seq_len = logits.size(1)
        if ingredients_mask.size(1) > seq_len:
            ingredients_mask = ingredients_mask[:, :seq_len]

        # Create point clouds for both the prediction and the ground truth
        point_clouds_pred = self._create_point_cloud(logits, ingredients_mask, embedding_matrix)
        point_clouds_gt = self._create_point_cloud(labels, ingredients_mask, embedding_matrix)

        # Calculate loss for each item in the batch and average the results
        batch_losses = []
        for pc_pred, pc_gt in zip(point_clouds_pred, point_clouds_gt):
            # Only compute loss if both point clouds have points and have the same number of points
            if pc_pred.numel() > 0 and pc_pred.shape[0] == pc_gt.shape[0]:
                # Unsqueeze to add a batch dimension of 1 for the loss function
                loss = self.loss_fn(pc_pred.unsqueeze(0), pc_gt.unsqueeze(0))
                batch_losses.append(loss.mean())

        if not batch_losses:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
            
        return torch.stack(batch_losses).mean()

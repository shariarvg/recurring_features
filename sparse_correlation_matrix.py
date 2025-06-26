from central_imports import *

class SparseCorrelationMatrix:
    """Class for computing and storing sparse correlations between features."""
    def __init__(self, threshold: float, k: int):
        self.threshold = threshold
        self.k_activating_examples = k
        self.nonzeros = []
        self.processed_blocks = set()  # Keep track of which blocks we've processed

    def compute_activation_mask(self, tensor: t.Tensor) -> t.Tensor:
        """Compute binary mask of where features are activated (non-zero)."""
        return (tensor != 0).float()

    def compute_conditional_probabilities(
        self,
        mask1: t.Tensor,
        mask2: t.Tensor,
        i: int,
        j: int
    ) -> Tuple[float, float]:
        """Compute P(i|j) and P(j|i) for features i and j."""
        # Get activation masks for features i and j
        act_i = mask1[:, i]
        act_j = mask2[:, j]
        
        # Compute P(i|j) = P(i and j) / P(j)
        p_i_given_j = (act_i * act_j).sum() / (act_j.sum() + 1e-8)
        
        # Compute P(j|i) = P(i and j) / P(i)
        p_j_given_i = (act_i * act_j).sum() / (act_i.sum() + 1e-8)
        
        return float(p_i_given_j), float(p_j_given_i)

    def process_block(
        self,
        T1: t.Tensor,
        T2: t.Tensor,
        start_i: int,
        end_i: int,
        start_j: int,
        end_j: int
    ) -> None:
        """
        Process a block of the correlation matrix.
        
        Args:
            T1: First tensor of shape (B, D1)
            T2: Second tensor of shape (B, D2)
            start_i, end_i: Range of indices for first dimension
            start_j, end_j: Range of indices for second dimension
        """
        if T1.ndim == 3:
            T1 = T1.mean(dim=1)
        if T2.ndim == 3:
            T2 = T2.mean(dim=1)

        # Get the relevant slices
        T1_slice = T1[:, start_i:end_i]
        T2_slice = T2[:, start_j:end_j]

        # Center and normalize the data
        T1_centered = T1_slice - T1_slice.mean(dim=0, keepdim=True)
        T2_centered = T2_slice - T2_slice.mean(dim=0, keepdim=True)

        T1_norm = T1_centered.norm(dim=0, keepdim=True) + 1e-8
        T2_norm = T2_centered.norm(dim=0, keepdim=True) + 1e-8

        T1_normalized = T1_centered / T1_norm
        T2_normalized = T2_centered / T2_norm

        # Compute correlation matrix for this block
        corr_matrix = T1_normalized.T @ T2_normalized

        # Get indices where |corr| > threshold
        mask = (corr_matrix.abs() > self.threshold)
        coords = mask.nonzero(as_tuple=False)

        # Compute activation masks for conditional probabilities
        mask1 = self.compute_activation_mask(T1)
        mask2 = self.compute_activation_mask(T2)

        # Process each significant correlation in this block
        for i, j in coords:
            global_i = int(i) + start_i
            global_j = int(j) + start_j
            corr = float(corr_matrix[i, j])
            
            # Skip if we've already processed this pair
            if (global_i, global_j) in self.processed_blocks:
                continue
                
            p_i_given_j, p_j_given_i = self.compute_conditional_probabilities(
                mask1, mask2, global_i, global_j
            )
            
            self.nonzeros.append((
                global_i, global_j, corr,
                p_i_given_j, p_j_given_i,
                tuple(t.topk(T1_centered[:,i], self.k_activating_examples).indices.tolist()),
                tuple(t.topk(T2_centered[:,j], self.k_activating_examples).indices.tolist())
            ))
            self.processed_blocks.add((global_i, global_j))

    def save(self, savename: str) -> None:
        """Save correlation results to a pickle file."""
        columns = ['i', 'j', 'corr', 'p_i_given_j', 'p_j_given_i', 'topk_i', 'topk_j']
        df = pd.DataFrame(self.nonzeros, columns=columns)
        df.to_pickle(savename)

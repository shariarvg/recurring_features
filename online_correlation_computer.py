from central_imports import *

class OnlineCorrelationComputer:
    """Class for computing correlations in an online fashion."""
    def __init__(self, D1: int, D2: int, threshold: float, k: int):
        self.D1 = D1
        self.D2 = D2
        self.threshold = threshold
        self.k = k
        
        # Initialize running statistics
        self.n = 0  # number of samples
        self.sum1 = t.zeros(D1, device='cpu')  # sum of first features
        self.sum2 = t.zeros(D2, device='cpu')  # sum of second features
        self.sum_sq1 = t.zeros(D1, device='cpu')  # sum of squares of first features
        self.sum_sq2 = t.zeros(D2, device='cpu')  # sum of squares of second features
        self.sum_prod = t.zeros((D1, D2), device='cpu')  # sum of products
        
        # For tracking top k examples
        self.top_k_values1 = t.full((D1, k), float('-inf'), device='cpu')
        self.top_k_values2 = t.full((D2, k), float('-inf'), device='cpu')
        self.top_k_indices1 = t.zeros((D1, k), dtype=t.long, device='cpu')
        self.top_k_indices2 = t.zeros((D2, k), dtype=t.long, device='cpu')
        
        # For tracking activation masks
        self.activation_mask1 = t.zeros(D1, device='cpu')
        self.activation_mask2 = t.zeros(D2, device='cpu')
        self.joint_activation = t.zeros((D1, D2), device = 'cpu')
        
        self.nonzeros = []
        self.processed_blocks = set()

    def update(self, batch1: t.Tensor, batch2: t.Tensor, batch_idx: int) -> None:
        """
        Update running statistics with a new batch of data.

        Args:
            batch1: Tensor of shape (B, L, D1) or (B, D1)
            batch2: Tensor of shape (B, L, D2) or (B, D2)
            batch_idx: Index of the first element in this batch
        """
        t0 = time.time()

        batch1 = batch1.cpu()
        batch2 = batch2.cpu()
        if batch1.ndim == 3:
            batch1 = batch1.mean(dim=1)
        if batch2.ndim == 3:
            batch2 = batch2.mean(dim=1)
        B = batch1.shape[0]
        print(f"[{batch_idx}] Preprocessing done in {time.time() - t0:.3f}s", flush=True)

        t1 = time.time()
        self.sum1 += batch1.sum(dim=0)
        self.sum2 += batch2.sum(dim=0)
        self.sum_sq1 += (batch1 ** 2).sum(dim=0)
        self.sum_sq2 += (batch2 ** 2).sum(dim=0)
        print(f"[{batch_idx}] Sums updated in {time.time() - t1:.3f}s", flush=True)

        t2 = time.time()
        self.sum_prod += batch1.T @ batch2
        self.joint_activation += t.einsum("bi,bj->ij", (batch1 > 0).float(), (batch2 > 0).float())
        print(f"[{batch_idx}] Product and joint activation updated in {time.time() - t2:.3f}s", flush=True)

        t3 = time.time()
        self.activation_mask1 += (batch1 != 0).sum(dim=0)
        self.activation_mask2 += (batch2 != 0).sum(dim=0)
        print(f"[{batch_idx}] Activation masks updated in {time.time() - t3:.3f}s", flush=True)

        t4 = time.time()
        values1, indices1 = t.topk(batch1.T, k=min(self.k, B), dim=1)
        values2, indices2 = t.topk(batch2.T, k=min(self.k, B), dim=1)
        global_indices1 = indices1 + batch_idx
        global_indices2 = indices2 + batch_idx
        print(f"[{batch_idx}] Top-k extraction done in {time.time() - t4:.3f}s", flush=True)

        t5 = time.time()
        combined_vals1 = t.cat([self.top_k_values1, values1], dim=1)
        combined_inds1 = t.cat([self.top_k_indices1, global_indices1], dim=1)
        combined_vals2 = t.cat([self.top_k_values2, values2], dim=1)
        combined_inds2 = t.cat([self.top_k_indices2, global_indices2], dim=1)

        top_vals1, top_idx1 = t.topk(combined_vals1, self.k, dim=1)
        top_vals2, top_idx2 = t.topk(combined_vals2, self.k, dim=1)

        row_indices1 = t.arange(self.D1).unsqueeze(1)
        self.top_k_values1 = top_vals1
        self.top_k_indices1 = combined_inds1[row_indices1, top_idx1]

        row_indices2 = t.arange(self.D2).unsqueeze(1)
        self.top_k_values2 = top_vals2
        self.top_k_indices2 = combined_inds2[row_indices2, top_idx2]
        print(f"[{batch_idx}] Top-k merge done in {time.time() - t5:.3f}s", flush=True)

        self.n += B
        print(f"[{batch_idx}] Total update time: {time.time() - t0:.3f}s\n", flush=True)

        
    def filter_coords(self, threshold):
        mask = self.joint_activation / (self.activation_mask1[:,None]+self.activation_mask2[None,:]) > threshold
        i_coord, j_coord = t.nonzero(mask, as_tuple = True)
        return i_coord, j_coord
    
    def process_from_coords(self, i_coords, j_coords):
        mean1 = self.sum1[i_coords] / self.n
        mean2 = self.sum2[j_coords] / self.n
        std1 = t.sqrt(self.sum_sq1[i_coords] / self.n - mean1 ** 2)
        std2 = t.sqrt(self.sum_sq2[j_coords] / self.n - mean2 ** 2)
        
        sum_prod = self.sum_prod[i_coords, j_coords]
        
        corrs = (sum_prod / self.n - mean1.unsqueeze(1) * mean2.unsqueeze(0)) / (
            std1 * std2 + 1e-8
        )
        
        mask = (corr_matrix.abs() > self.threshold)
        coords_df = mask.nonzero(as_tuple=False)
        
        for (i, j, corrs) in zip(i_coords, j_coords, corrs):
            global_i = i_coords[i]
            global_j = j_coords[j]
            
            # Skip if we've already processed this pair
            if (global_i, global_j) in self.processed_blocks:
                continue
            
            corr = float(corr_matrix[i, j])
            
            p_i_given_j = (self.joint_activation[global_i, global_j] / self.activation_mask2[global_j]) if self.activation_mask2[global_j] > 0 else 0
            p_j_given_i = (self.joint_activation[global_i, global_j] / self.activation_mask1[global_i]) if self.activation_mask1[global_i] > 0 else 0

            topk_i = self.top_k_indices1[global_i].tolist()
            topk_j = self.top_k_indices2[global_j].tolist()

            self.nonzeros.append((
                global_i, global_j, corr,
                float(p_i_given_j), float(p_j_given_i),
                tuple(topk_i),
                tuple(topk_j)
            ))
            self.processed_blocks.add((global_i, global_j))
            

    def process_block(
        self,
        start_i: int,
        end_i: int,
        start_j: int,
        end_j: int
    ) -> None:
        """Process a block of the correlation matrix using accumulated statistics."""
        # Compute means
        mean1 = self.sum1 / self.n
        mean2 = self.sum2 / self.n
        
        # Compute standard deviations
        std1 = t.sqrt(self.sum_sq1 / self.n - mean1 ** 2)
        std2 = t.sqrt(self.sum_sq2 / self.n - mean2 ** 2)
        
        # Get the relevant slices
        mean1_slice = mean1[start_i:end_i]
        mean2_slice = mean2[start_j:end_j]
        std1_slice = std1[start_i:end_i]
        std2_slice = std2[start_j:end_j]
        sum_prod_slice = self.sum_prod[start_i:end_i, start_j:end_j]
        
        # Compute correlation matrix for this block
        corrs = (sum_prod_slice / self.n - mean1_slice.unsqueeze(1) * mean2_slice.unsqueeze(0)) / (
            std1_slice.unsqueeze(1) * std2_slice.unsqueeze(0) + 1e-8
        )
        
        # Get indices where |corr| > threshold
        mask = (corr_matrix.abs() > self.threshold)
        coords = mask.nonzero(as_tuple=False)
        
        # Process each significant correlation in this block
        for (i, j) in coords:
            global_i = int(i) + start_i
            global_j = int(j) + start_j
            
            # Skip if we've already processed this pair
            if (global_i, global_j) in self.processed_blocks:
                continue
            
            corr = float(corr_matrix[i, j])
            
            # Compute conditional probabilities
            p_i_given_j = (self.joint_activation[global_i, global_j] / self.activation_mask2[global_j]) if self.activation_mask2[global_j] > 0 else 0
            p_j_given_i = (self.joint_activation[global_i, global_j] / self.activation_mask1[global_i]) if self.activation_mask1[global_i] > 0 else 0
            
            # Convert indices to list before creating tuple
            topk_i = self.top_k_indices1[global_i].tolist()
            topk_j = self.top_k_indices2[global_j].tolist()
            
            self.nonzeros.append((
                global_i, global_j, corr,
                float(p_i_given_j), float(p_j_given_i),
                tuple(topk_i),
                tuple(topk_j)
            ))
            self.processed_blocks.add((global_i, global_j))

    def save(self, savename: str) -> None:
        """Save correlation results to a pickle file."""
        columns = ['i', 'j', 'corr', 'p_i_given_j', 'p_j_given_i', 'topk_i', 'topk_j']
        df = pd.DataFrame(self.nonzeros, columns=columns)
        df.to_pickle(savename)
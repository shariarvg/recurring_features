import transformer_lens
import sae_lens

import gc
import itertools
import math
import os
import random
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Literal, TypeAlias, Optional, Tuple, List, Dict, Union

import time

import circuitsvis as cv
import einops
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import torch as t
from datasets import load_dataset, Dataset
from huggingface_hub import hf_hub_download
from IPython.display import HTML, IFrame, clear_output, display
from jaxtyping import Float, Int
from openai import OpenAI
from rich import print as rprint
from rich.table import Table
from sae_lens import (
    SAE,
    ActivationsStore,
    HookedSAETransformer,
    LanguageModelSAERunnerConfig,
    SAEConfig,
    SAETrainingRunner,
    upload_saes_to_huggingface,
)
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
from sae_vis import SaeVisConfig, SaeVisData, SaeVisLayoutConfig
from tabulate import tabulate
from torch import Tensor, nn
from torch.distributions.categorical import Categorical
from torch.nn import functional as F
from tqdm.auto import tqdm
from transformer_lens import ActivationCache, HookedTransformer
from transformer_lens.hook_points import HookPoint
from transformer_lens.utils import get_act_name, test_prompt, to_numpy
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def set_all_seeds(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed(seed)
    t.backends.cudnn.deterministic = True
    t.backends.cudnn.benchmark = True

class TokenizedDataset(Dataset):
    """Dataset class for tokenizing text data."""
    def __init__(self, dataset: Dataset, tokenizer: GPT2Tokenizer, max_length: int):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, t.Tensor]:
        text = self.dataset[idx]["text"]
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors="pt"
        )
        return {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"]
        }

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

def load_model_and_saes(
    device: str,
    release: str,
    layer1: int,
    layer2: int
) -> Tuple[HookedSAETransformer, SAE, SAE]:
    """Load the GPT2 model and SAEs for specified layers."""
    model = HookedSAETransformer.from_pretrained("gpt2-small", device=device)
    
    sae1, cfg_dict1, sparsity1 = SAE.from_pretrained(
        release=release,
        sae_id=f"blocks.{layer1}.hook_resid_pre",
        device=str(device),
    )
    
    sae2, cfg_dict2, sparsity2 = SAE.from_pretrained(
        release=release,
        sae_id=f"blocks.{layer2}.hook_resid_pre",
        device=str(device),
    )
    
    return model, sae1, sae2

def process_dataset(
    dataset_name: str,
    dataset_config: Optional[str] = None,
    subset_size: int = 5000,
    batch_size: int = 4,
    max_length: int = 20,
    seed: int = 0
) -> DataLoader:
    """Process and tokenize the dataset."""
    # Load dataset
    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config, split="train")
    else:
        dataset = load_dataset(dataset_name, split="train")
    
    # Create subset
    subset = dataset.shuffle(seed=seed).select(range(subset_size))
    
    # Setup tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataloader
    tokenized_dataset = TokenizedDataset(subset, tokenizer, max_length)
    return DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=False)
import time
import torch as t

def compute_correlations(
    model: HookedSAETransformer,
    sae1: SAE,
    sae2: SAE,
    dataloader: DataLoader,
    device: str,
    correlation_threshold: float = 0.9,
    k_examples: int = 3,
    output_file: str = "correlated_features.pkl",
    block_size: int = 1000,
    save_frequency: int = 10,
    online: bool = False
) -> None:
    """
    Compute correlations between features from two layers.
    """
    if online:
        print("Initializing online correlation computer...", flush = True)
        init_time = time.time()
        corr_computer = OnlineCorrelationComputer(
            sae1.cfg.d_sae,
            sae2.cfg.d_sae,
            correlation_threshold,
            k_examples
        )
        print(f"Initialized in {time.time() - init_time:.2f} seconds\n", flush = True)

        print("Computing correlations online over dataloader...", flush = True)
        batch_idx = 0
        total_batch_time = 0
        for batch in dataloader:
            start_time = time.time()
            with t.no_grad():
                layerwise_activations_cache = model.run_with_cache(
                    batch['input_ids'].to(device),
                    names_filter=[sae1.cfg.hook_name, sae2.cfg.hook_name],
                    stop_at_layer=max(sae1.cfg.hook_layer, sae2.cfg.hook_layer) + 1,
                    prepend_bos=False,
                )[1]
                if batch_idx == 0:
                    s = time.time()
                act1 = sae1.encode(layerwise_activations_cache[sae1.cfg.hook_name])
                act2 = sae2.encode(layerwise_activations_cache[sae2.cfg.hook_name])
                
                if batch_idx == 0:
                    e = time.time()
                    print(f"Encoding first batch with two sae's takes {(e-s):.2f} seconds", flush = True)
                
                corr_computer.update(act1, act2, batch_idx)
                batch_idx += act1.shape[0]
            elapsed = time.time() - start_time
            total_batch_time += elapsed
            if batch_idx % 1000 < act1.shape[0]:
                print(f"Processed {batch_idx} examples so far, avg time per batch: {total_batch_time / (batch_idx / act1.shape[0]):.2f}s", flush = True)

        print(f"\nFinished dataloader pass in {total_batch_time:.2f} seconds\n", flush = True)

        # Block processing
        print("Starting block-wise correlation processing...")
        D1 = sae1.cfg.d_sae
        D2 = sae2.cfg.d_sae
        total_blocks = ((D1 + block_size - 1) // block_size) * ((D2 + block_size - 1) // block_size)
        block_count = 0
        block_start_time = time.time()
        
        ic, jc = corr_computer.filter_coords(0.1)
        corr_computer.process_from_coords(ic, jc)

        '''
        for start_i in range(0, D1, block_size):
            end_i = min(start_i + block_size, D1)
            for start_j in range(0, D2, block_size):
                end_j = min(start_j + block_size, D2)
                tic = time.time()

                

                toc = time.time()
                block_count += 1
                print(f"Processed block {block_count}/{total_blocks} in {toc - tic:.2f}s")

                if block_count % save_frequency == 0:
                    print(f"Saving intermediate results at block {block_count}...")
                    save_time = time.time()
                    corr_computer.save(output_file)
                    print(f"Saved in {time.time() - save_time:.2f} seconds")
        '''
        print(f"\nFinished filtered processing in {time.time() - block_start_time:.2f} seconds\n", flush = True)

        # Final save
        print("Saving final results...")
        final_save_time = time.time()
        corr_computer.save(output_file)
        print(f"Final save done in {time.time() - final_save_time:.2f} seconds\n")

        
    else:
        # Original batch processing method
        scm = SparseCorrelationMatrix(correlation_threshold, k_examples)
        
        # Process activations in batches
        sparse_codes = defaultdict(list)
        
        print("Collecting activations...")
        with t.no_grad():
            for batch in dataloader:
                layerwise_activations_cache = model.run_with_cache(
                    batch['input_ids'].to(device),
                    names_filter=[sae1.cfg.hook_name, sae2.cfg.hook_name],
                    stop_at_layer=max(sae1.cfg.hook_layer, sae2.cfg.hook_layer) + 1,
                    prepend_bos=False,
                )[1]

                for sae in [sae1, sae2]:
                    sparse_act = sae.encode(layerwise_activations_cache[sae.cfg.hook_name])
                    sparse_codes[sae.cfg.hook_name].append(sparse_act)
        
        # Concatenate codes
        codes1 = t.cat(sparse_codes[sae1.cfg.hook_name], dim=0)
        codes2 = t.cat(sparse_codes[sae2.cfg.hook_name], dim=0)
        
        # Process in blocks
        D1 = sae1.cfg.d_sae
        D2 = sae2.cfg.d_sae
        total_blocks = ((D1 + block_size - 1) // block_size) * ((D2 + block_size - 1) // block_size)
        block_count = 0
        
        print(f"Processing {total_blocks} blocks...")
        for start_i in range(0, D1, block_size):
            end_i = min(start_i + block_size, D1)
            for start_j in range(0, D2, block_size):
                end_j = min(start_j + block_size, D2)
                
                print(f"Processing block {block_count + 1}/{total_blocks}")
                scm.process_block(codes1, codes2, start_i, end_i, start_j, end_j)
                
                block_count += 1
                if block_count % save_frequency == 0:
                    print(f"Saving intermediate results...")
                    scm.save(output_file)
        
        # Final save
        print("Saving final results...")
        scm.save(output_file)

def main(
    dataset_name: str,
    dataset_config: Optional[str] = None,
    layer1: int = 7,
    layer2: int = 8,
    subset_size: int = 5000,
    batch_size: int = 8,
    max_length: int = 20,
    correlation_threshold: float = 0.9,
    k_examples: int = 3,
    device: str = 'cuda',
    release: str = "gpt2-small-res-jb",
    seed: int = 0
) -> None:
    """Main function to run the correlation analysis."""
    set_all_seeds(seed)
    
    # Load model and SAEs
    model, sae1, sae2 = load_model_and_saes(device, release, layer1, layer2)
    
    # Process dataset
    dataloader = process_dataset(
        dataset_name,
        dataset_config,
        subset_size,
        batch_size,
        max_length,
        seed
    )
    
    # Compute correlations
    output_file = f"TEST_correlated_features_{layer1}_{layer2}_{dataset_name}_{subset_size}.pkl"
    compute_correlations(
        model,
        sae1,
        sae2,
        dataloader,
        device,
        correlation_threshold,
        k_examples,
        output_file,
        online=True
    )

if __name__ == "__main__":
    # Example usage
    main(
        dataset_name="wikitext",
        dataset_config="wikitext-103-raw-v1",
        layer1=2,
        layer2=11,
        subset_size=100
    )

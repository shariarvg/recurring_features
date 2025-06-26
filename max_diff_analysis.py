import torch as t
import pandas as pd
from sae_lens import SAE, HookedSAETransformer
from typing import Tuple, Optional
from processing_and_loading import process_dataset
import numpy as np

def get_all_caches(dataloader, model: HookedSAETransformer, sae1: SAE, sae2: SAE):
    '''
    Getting relevant caches at once, so we can iterate over each latent one at a time later
    '''
    out_l1 = []
    out_l2 = []
    for batch in dataloader:
        batch_cache = get_single_cache(batch, model, sae1, sae2)
        out_l1.append(cache[sae1.cfg.hook_name])
        out_l2.append(cache[sae2.cfg.hook_name])
    return {sae1.cfg.hook_name: t.cat(out_l1, axis = 0), sae2.cfg.hook_name: t.cat(out_l2, axis = 0)}

def get_single_cache(batch, model, sae1, sae2):
    batch_cache = model.run_with_cache(
            batch['input_ids'],
            names_filter=[sae1.cfg.hook_name, sae2.cfg.hook_name],
            stop_at_layer=max(sae1.cfg.hook_layer, sae2.cfg.hook_layer) + 1,
            prepend_bos=False,
        )[1]
    return batch_cache

def single_latent_encode(vec, i: int, sae1: SAE):
    return t.relu((vec - sae1.b_dec[i]) @ sae1.W_enc[:,i] + sae1.b_enc[i])

def multi_latent_encode(vec, all_i, sae1: SAE):
    
    diff = vec - sae1.b_dec
    prod = diff @ sae1.W_enc[:,all_i]
    
    return t.relu(prod + sae1.b_enc[all_i])

def get_activations_diff(cache, model: HookedSAETransformer, sae1: SAE, sae2: SAE, latent_pairs, summarize = False):
    '''
    Takes both caches (stored in cache), a single latent pair, and obtains the difference in position
    '''
    # Get activations for both latents
    act1 = multi_latent_encode(cache[sae1.cfg.hook_name], latent_pairs[:,0], sae1).detach().cpu()
    act2 = multi_latent_encode(cache[sae2.cfg.hook_name], latent_pairs[:,1], sae2).detach().cpu()
    
    act1_max_seq = t.argmax(act1, axis=1)
    act2_max_seq = t.argmax(act2, axis=1)
    
    mask = ((t.sum(act1, axis=1) > 0) & (t.sum(act2, axis=1) > 0))
    max_diff_full = (act2_max_seq - act1_max_seq).float()  # (batch_size, num_latents)
    max_diff_masked = t.where(mask, max_diff_full, t.tensor(float('nan')))  # (batch_size, num_latents)
    
    if not summarize:
        return max_diff_masked
    
    mean_diff = t.nanmean(max_diff_masked, dim=0)  # (num_latents,)
    std_diff = np.nanstd(max_diff_masked.numpy(), axis=0)    # (num_latents,)
    count_valid = mask.sum(dim=0)                # (num_latents,)

    return mean_diff, std_diff, count_valid

def run_analysis_by_batch(dataloader, model, sae1, sae2, latent_pairs):
    all_max_diff = []
    for batch in dataloader:
        batch_cache = get_single_cache(batch, model, sae1, sae2)
        max_diff_masked = get_activations_diff(batch_cache, model, sae1, sae2, latent_pairs, False).numpy()
        all_max_diff.append(max_diff_masked)
        
    all_max_diff = np.concatenate(all_max_diff, axis = 0)
    
    mean_diff = np.nanmean(all_max_diff, axis=0)  # (num_latents,)
    std_diff = np.nanstd(all_max_diff, axis=0)    # (num_latents,)
    count_valid = (~np.isnan(all_max_diff)).sum(axis = 0)

    return mean_diff, std_diff, count_valid

def run_analysis_all_at_once(dataloader, model, sae1, sae2, latent_pairs):
    cache = get_all_caches(dataloader, model, sae1, sae2)
    return get_activations_diff(cache, model, sae1, sae2, latent_pairs, True)

def max_diff_analysis(
    prompts,
    model: HookedSAETransformer,
    sae1: SAE,
    sae2: SAE,
    fname: str,
    n_latent_pairs: int = 100,
    output_prefix: str = "diff_"
) -> None:
    '''
    Analyze the maximum difference in activation positions between pairs of latents.
    
    Args:
        prompts: Input prompts to analyze
        model: The transformer model
        sae1: First layer's SAE
        sae2: Second layer's SAE
        fname: Path to the pickle file containing latent pairs
        n_latent_pairs: Number of latent pairs to analyze
        output_prefix: Prefix for the output file name
    '''
    df = pd.read_pickle(fname)
    latent_pairs = df[['i', 'j']].iloc[:n_latent_pairs].to_numpy()
    means, stds, counts = run_analysis_by_batch(prompts, model, sae1, sae2, latent_pairs)
    
    print(means.shape)
    print(stds.shape)
    print(counts.shape)
    
    # Update dataframe with results
    df = pd.DataFrame()
    df['mean'] = means
    df['std'] = stds
    df['count'] = counts
    df['i'] = latent_pairs[:,0]
    df['j'] = latent_pairs[:,1]
    
    # Save results
    output_file = f"{output_prefix}{fname}"
    df.to_pickle(output_file)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    # Example usage
    from sae_lens import HookedSAETransformer, SAE
    
    # Load model and SAEs
    device = "cuda"
    model = HookedSAETransformer.from_pretrained("gpt2-small", device=device)
    
    sae1, cfg_dict1, sparsity1 = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id="blocks.2.hook_resid_pre",
        device=str(device),
    )
    
    sae2, cfg_dict2, sparsity2 = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id="blocks.11.hook_resid_pre",
        device=str(device),
    )
    
    # Example prompts
    loader = process_dataset(
        dataset_name="wikitext",
        dataset_config="wikitext-103-raw-v1",
        subset_size=50_000,
        batch_size=8,
        max_length=100,
    ) 
    
    
    # Run analysis
    max_diff_analysis(
        prompts=loader,
        model=model,
        sae1=sae1,
        sae2=sae2,
        fname="correlated_features_2_11_wikitext_50000_0.pkl",
        n_latent_pairs=100
    ) 
from central_imports import *
from tokenized_dataset import TokenizedDataset
from sparse_correlation_matrix import SparseCorrelationMatrix


def set_all_seeds(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed(seed)
    t.backends.cudnn.deterministic = True
    t.backends.cudnn.benchmark = True

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

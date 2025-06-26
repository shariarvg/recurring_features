from central_imports import *
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


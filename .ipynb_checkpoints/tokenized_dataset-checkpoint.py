from central_imports import *

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
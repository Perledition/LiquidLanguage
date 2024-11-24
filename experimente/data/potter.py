import os
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from typing import Optional

class HarryPotterDataset(Dataset):
    def __init__(self, file_path: str, seq_length: int, device, tokenizer_path: Optional[str] = None):
        with open(file_path, "r", encoding="utf-8") as file:
            self.text = file.read()
        
        self.device = device
        self.seq_length = seq_length
        special_tokens=["[SOS]", "[EOS]", "[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
        # Initialize or load tokenizer
        if tokenizer_path and os.path.exists(tokenizer_path):
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
        else:
            # Create and train a new BPE tokenizer with explicit special tokens
            self.tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
            
            # Configure tokenizer with special tokens
            self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
            
            # Train the tokenizer with special tokens configuration
            trainer = trainers.BpeTrainer(
                vocab_size=50308,
                special_tokens=special_tokens,
                initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                show_progress=True
            )

            self.tokenizer.train_from_iterator([self.text], trainer=trainer)
            self.tokenizer.add_special_tokens(special_tokens)       
            
            # Save the tokenizer if path is provided
            if tokenizer_path:
                self.tokenizer.save(tokenizer_path)
        
        # Get and store special token IDs
        print(f"Pad token ID: {self.tokenizer.token_to_id('[UNK]')}")
        
        # Verify special tokens are properly set
        if any(self.tokenizer.token_to_id(token_id) is None for token_id in special_tokens):
            raise ValueError("Special tokens were not properly initialized in the tokenizer")
        
        # Tokenize the entire text
        encoded = self.tokenizer.encode(self.text)
        self.data = encoded.ids

        self.sos_id = self.tokenizer.token_to_id("[SOS]")
        self.eos_id = self.tokenizer.token_to_id("[EOS]")
        self.pad_id = self.tokenizer.token_to_id("[PAD]")
        self.unk_id = self.tokenizer.token_to_id("[UNK]")
        
    def __len__(self):
        return len(self.data) - self.seq_length - 1  # -1 for target token
        
    def __getitem__(self, idx):
        # Get sequence with room for SOS and target
        sequence = self.data[idx:idx + self.seq_length - 1]  # -1 to make room for SOS
        target = self.data[idx + self.seq_length - 1]  # Next token is target

        # create a one hot encoding for the target token
        #target_vector = torch.zeros(self.vocab_size)
        #target_vector[target] = 1
        
        # Add SOS token at the beginning
        input_sequence = [self.sos_id] + sequence
        
        # Pad sequence if necessary
        if len(input_sequence) < self.seq_length:
            input_sequence = input_sequence + [self.pad_id] * (self.seq_length - len(input_sequence))
        
        input_sequence = [value if value != None else self.unk_id for value in input_sequence]

        return (
            torch.tensor(input_sequence, dtype=torch.long).to(self.device),
            # torch.tensor(target_vector, dtype=torch.long).to(self.device)
            torch.tensor(target, dtype=torch.long).to(self.device)
        )
    
    def encode(self, text):
        """Helper method to encode text to token ids"""
        return self.tokenizer.encode(text).ids
    
    def decode(self, token_ids):
        """Helper method to decode token ids back to text"""
        return self.tokenizer.decode(token_ids)
    
    @property
    def vocab_size(self):
        """Return the vocabulary size"""
        return self.tokenizer.get_vocab_size()

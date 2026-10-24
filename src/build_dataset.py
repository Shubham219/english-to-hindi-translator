import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from torch.utils.data import DataLoader
from tokenizers import Tokenizer



class EnglishHindiDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        
        self.seq_len = seq_len
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("<s>")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("</s>")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("<pad>")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # Transform the text into tokens
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Add sos, eos and padding to each sentence
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # We will add <s> and </s>
        # We will only add <s>, and </s> only on the label
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        # Make sure the number of padding tokens is not negative. If it is, the sentence is too long
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        # Add <s> and </s> token
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add only <s> token
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add only </s> token
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Double check the size of the tensors to make sure they are all seq_len long
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            "label": label,  # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }



def get_ds():
    
    train_dataset = load_dataset("cfilt/iitb-english-hindi", split="train")
    valid_dataset = load_dataset("cfilt/iitb-english-hindi", split="validation")
    test_dataset  = load_dataset("cfilt/iitb-english-hindi", split="test")

    # Build tokenizers
    tokenizer_src = get_tokenizer(ln='hi')
    tokenizer_tgt = get_tokenizer(ln='hi')

    train_ds = EnglishHindiDataset(train_dataset, tokenizer_src, tokenizer_tgt, src_lang='en', tgt_lang='hi', seq_len=350)
    val_ds = EnglishHindiDataset(valid_dataset, tokenizer_src, tokenizer_tgt, src_lang='en', tgt_lang='hi', seq_len=350)
    test_ds = EnglishHindiDataset(test_dataset, tokenizer_src, tokenizer_tgt, src_lang='en', tgt_lang='hi', seq_len=350)
    
    train_dataloader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, test_dataloader, tokenizer_src, tokenizer_tgt

def get_tokenizer(ln, path=None):
    if ln=='hi':
        if path is None:
            path = r"D:\PythonProjects\Make_LLM\hi_to_en_transformer\tokenizer\hindi_tokenizer.json"
        tokenizer = Tokenizer.from_file(path)
        return tokenizer
    else:
        ln='en'
        # Use hugging face tokenizer for English
        tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
        return tokenizer

train_dataloader, val_dataloader, test_dataloader, tokenizer_src, tokenizer_tgt = get_ds()
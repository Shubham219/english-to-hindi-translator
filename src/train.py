import os
import torch
from torch.nn import functional as F
import torch.nn as nn

from model import Transformer
from model import Decoder
from build_dataset import get_ds

from datasets import load_dataset
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from tqdm import tqdm


def get_config():

    config = {
        "vocab_size": 100000,
        "n_contx": 500,
        "n_embd": 32,
        "d_model": 32,
        "n_head": 8,
        "n_layer": 2,
        "droprate": 0.3,
    }

    return config


def generate_text(model, src_txt, src_tokenizer, tgt_tokenizer, max_token_len=20):
    
    print("Translating text for source:", src_txt)
    model.eval()

    sos_idx = tgt_tokenizer.token_to_id('<s>')
    eos_idx = tgt_tokenizer.token_to_id('</s>')

    src_tokens = src_tokenizer.encode(src_txt).ids
    src_tokens = [src_tokenizer.token_to_id('<s>')] + src_tokens + [src_tokenizer.token_to_id('</s>')]
    src_tokens = torch.tensor(src_tokens, dtype=torch.long).unsqueeze(0)  # Add batch dimension
    print("Source Tokens Size:", src_tokens.shape)

    encoder_output = model.encoder(src_tokens)
    print("Encoder Output:", encoder_output.shape)

    decoder_input = torch.as_tensor([[sos_idx]], dtype=torch.long)  # Start with <s> token

    while True:
        with torch.no_grad():
            logits = model.decoder(decoder_input, encoder_output)
        print("Logits:", logits.shape)
        next_token_logits = logits[:, -1, :]
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.argmax(probs, dim=-1).unsqueeze(0)

        decoder_input = torch.cat([decoder_input, next_token], dim=1)
        print("Decoder Input:", decoder_input.shape)
       
        if next_token.item() == eos_idx or decoder_input.shape[1] >= max_token_len:
            break
    return decoder_input.squeeze().tolist()  # Remove batch dimension and convert to list



def evaluate_model(model, val_loader, loss_fn, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            target_output = batch['label'].to(device)

            output = model(encoder_input, decoder_input)
            loss = loss_fn(output.view(-1, output.size(-1)), target_output.view(-1))
            total_loss += loss.item()

    return total_loss / len(val_loader)


def train_model(model, config):

    device = 'cuda' if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Creating dir to save the model checkpoints
    os.makedirs("checkpoints", exist_ok=True)

    train_dataloader, val_dataloader, test_loader, tokenizer_src, tokenizer_tgt = get_ds()

    model = Transformer(**config)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id("<pad>")).to(device)

    total_loss = 0.0
    global_step = 0
    initial_epoch = 0
    epochs = 1

    for epoch in range(initial_epoch, epochs):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            target_output = batch['label'].to(device)

            optimizer.zero_grad()

            output = model(encoder_input, decoder_input)
            loss = loss_fn(output.view(-1, output.size(-1)), target_output.view(-1))
            
            loss.backward()

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item()

            global_step += 1
            break

        validation_loss = evaluate_model(model, val_dataloader, loss_fn, device)
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {total_loss/len(train_loader):.4f}, Validation Loss: {validation_loss:.4f}")   

        # save the model at the end of each epoch
        checkpoint_path = f"checkpoints/transformer_epoch_{epoch+1}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, checkpoint_path)

        return total_loss / len(train_loader)        


if __name__ == "__main__":
    config = get_config()
    model = Transformer(**config)
    
    train_dataloader, val_dataloader, test_loader, tokenizer_src, tokenizer_tgt = get_ds()

    train_model(model, config)

    out_tokens = generate_text(model, "This is a test", tokenizer_src, tokenizer_tgt, max_token_len=20)
    print(tokenizer_tgt.decode(out_tokens))

    eval_loss = evaluate_model(model, val_dataloader, torch.nn.CrossEntropyLoss(), device='cpu')
    print(f"Validation Loss: {eval_loss:.4f}")
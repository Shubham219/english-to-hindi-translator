import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_contx, n_embd, d_model, n_head, qkv_bias, is_causal, attn_dropout=0.0):
        super().__init__()
        if n_head <= 0:
            raise ValueError("n_head must be greater than 0")
        if d_model % n_head != 0:
            raise ValueError("d_model must be divisible by n_head")
        self.is_causal = is_causal
        self.d_model = d_model
        self.n_head = n_head
        # q, k, v into 1 c_attn
        self.wq = nn.Linear(n_embd, d_model, bias=qkv_bias)
        self.wk = nn.Linear(n_embd, d_model, bias=qkv_bias)
        self.wv = nn.Linear(n_embd, d_model, bias=qkv_bias)
        self.c_proj = nn.Linear(d_model, d_model, bias=True)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        self.causal_mask: torch.Tensor
        self.register_buffer('causal_mask',
                             torch.triu(torch.ones(n_contx, n_contx, dtype=torch.bool), diagonal=1)
                             )  

    def forward(self, xq, xk, xv):
        if xq.shape[0] != xk.shape[0] or xk.shape[:2] != xv.shape[:2]:
            raise ValueError("Query, key, and value batch/sequence dimensions are incompatible")

        head_dim = self.d_model // self.n_head
        q = self.wq(xq).reshape(
            xq.shape[0], xq.shape[1], self.n_head, head_dim
        ).transpose(1, 2)
        k = self.wk(xk).reshape(
            xk.shape[0], xk.shape[1], self.n_head, head_dim
        ).transpose(1, 2)
        v = self.wv(xv).reshape(
            xv.shape[0], xv.shape[1], self.n_head, head_dim
        ).transpose(1, 2)
        if self.flash:
            attn = torch.nn.functional.scaled_dot_product_attention(query=q, key=k, value=v, 
                                                                 attn_mask=None, 
                                                                 dropout_p=self.attn_dropout.p if self.training else 0.0, 
                                                                 is_causal=self.is_causal, 
                                                                 scale=None, 
                                                                 enable_gqa=False
                                                                )
        
        else:
            wei = (q @ k.transpose(2, 3)) / (head_dim ** 0.5)

            if self.is_causal:
                target_len = xq.shape[1]
                mask_bool = self.causal_mask[:target_len, :xk.shape[1]]
                wei.masked_fill_(mask_bool, -torch.inf)
            
            attn_score = torch.softmax(wei, dim=-1)
            attn_score = self.attn_dropout(attn_score)
            attn = attn_score @ v

            
        attn = attn.transpose(1, 2).contiguous().reshape(
            xq.shape[0], xq.shape[1], self.d_model
        )
        output = self.c_proj(attn)
        return output
    

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=True)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=True)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = torch.nn.functional.gelu(x)
        x = self.c_proj(x)
        return x

    
class EncoderBlock(nn.Module):
    def __init__(self, n_contx, n_embd, d_model, n_head, droprate, qkv_bias=True):
        super().__init__()
        if d_model != n_embd:
            raise ValueError("d_model must equal n_embd for residual connections")
        self.attn = MultiHeadAttention(
            n_contx, n_embd, d_model, n_head, qkv_bias, is_causal=False, attn_dropout=droprate
        )
        self.l_norm1 = nn.LayerNorm(n_embd)
        self.ffw = FeedForward(n_embd)
        self.l_norm2 = nn.LayerNorm(n_embd)
        self.dropout = nn.Dropout(droprate)

    def forward(self, x):
        x = self.l_norm1(x + self.dropout(self.attn(x,x,x)))
        x = self.l_norm2(x + self.dropout(self.ffw(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, n_contx, n_embd, d_model, n_head, n_layer, droprate):
        super().__init__()
        self.n_contx=n_contx
        self.embd = nn.Embedding(vocab_size, n_embd)
        self.pos = nn.Embedding(n_contx, n_embd)
        self.dropout = nn.Dropout(droprate)
        self.enc_block = nn.Sequential(*[EncoderBlock(n_contx, n_embd, d_model, n_head, droprate) for _ in range(n_layer)])
    
    def forward(self, x):
        emb_tok = self.embd(x)
        pos_tok = self.pos(
            torch.arange(x.shape[1], dtype=torch.long, device=x.device)).unsqueeze(0)
        x = emb_tok+pos_tok
        x = self.dropout(x)
        x = self.enc_block(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, n_contx, n_embd, n_head, qkv_bias, d_model, droprate):
        super().__init__()
        if d_model != n_embd:
            raise ValueError("d_model must equal n_embd for residual connections")
        self.c_attn = MultiHeadAttention(
            n_contx, n_embd, d_model, n_head, qkv_bias, is_causal=True, attn_dropout=droprate
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(
            n_contx, n_embd, d_model, n_head, qkv_bias, is_causal=False, attn_dropout=droprate
        )
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.ffw = FeedForward(n_embd)
        self.layer_norm3 = nn.LayerNorm(n_embd)
        self.dropout1 = nn.Dropout(droprate)
        self.dropout2 = nn.Dropout(droprate)
        self.dropout3 = nn.Dropout(droprate)

    def forward(self, x, encoder_output):
        x = self.layer_norm1(x + self.dropout1(self.c_attn(x,x,x)))
        x = self.layer_norm2(x + self.dropout2(self.cross_attn(x, encoder_output, encoder_output)))
        x = self.layer_norm3(x + self.dropout3(self.ffw(x)))
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, n_embd, n_contx, n_head, qkv_bias, d_model, n_layer, droprate):
        super().__init__()
        self.embd = nn.Embedding(vocab_size, n_embd)
        self.pos = nn.Embedding(n_contx, n_embd)
        self.dropout1 = nn.Dropout(droprate)
        self.dec_block = nn.ModuleList([
            DecoderBlock(n_contx, n_embd, n_head, qkv_bias, d_model, droprate) for _ in range(n_layer)])
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x, encoder_output):
        emb_tok = self.embd(x)
        pos_tok = self.pos(torch.arange(0, x.shape[1], dtype=torch.long, device=x.device)).unsqueeze(0)
        x = self.dropout1(emb_tok + pos_tok)
        for block in self.dec_block:
            x = block(x, encoder_output)
        logits = self.lm_head(x)
        return logits


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_contx,
        n_embd,
        d_model,
        n_head,
        n_layer,
        droprate,
        qkv_bias=True,
        tgt_vocab_size=None,
    ):
        super().__init__()
        self.n_contx = n_contx
        self.encoder = Encoder(
            vocab_size, n_contx, n_embd, d_model, n_head, n_layer, droprate
        )
        self.decoder = Decoder(
            vocab_size if tgt_vocab_size is None else tgt_vocab_size,
            n_embd,
            n_contx,
            n_head,
            qkv_bias,
            d_model,
            n_layer,
            droprate,
        )

    def forward(self, encoder_input, decoder_input):
        if encoder_input.shape[1] > self.n_contx or decoder_input.shape[1] > self.n_contx:
            raise ValueError("Input sequence length exceeds the configured context length")

        encoder_output = self.encoder(encoder_input)
        return self.decoder(decoder_input, encoder_output)
    


if __name__ == "__main__":

    inputs = torch.randint(0, 8, (2, 8))
    model = Transformer(
        vocab_size=8,
        n_contx=8,
        n_embd=32,
        d_model=32,
        n_head=4,
        n_layer=2,
        droprate=0.3,
    )
    logits = model(inputs, inputs)
    print(logits.shape)

    # Encoder test
    enc_output = model.encoder(inputs)
    print(enc_output.shape)
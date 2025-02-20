import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self.max_len:
            pe = torch.zeros(seq_len, self.d_model)
            position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float) * (-math.log(10000.0) / self.d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)  # shape: (1, seq_len, d_model)
            return x + pe.to(x.device)
        return x + self.pe[:, :seq_len]

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, is_decoder=False):
        super().__init__()
        self.is_decoder = is_decoder
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout)
        if is_decoder:
            self.cross_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        if is_decoder:
            self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output=None, attn_mask=None, cross_attn_mask=None):
        # x: (batch, seq_len, d_model) → convert to (seq_len, batch, d_model)
        x_trans = x.transpose(0, 1)
        attn_output, _ = self.attention(x_trans, x_trans, x_trans, attn_mask=attn_mask)
        x_trans = self.norm1(x_trans + self.dropout(attn_output))
        
        if self.is_decoder and enc_output is not None:
            enc_output = enc_output.transpose(0, 1)
            cross_attn_output, _ = self.cross_attention(x_trans, enc_output, enc_output, attn_mask=cross_attn_mask)
            x_trans = self.norm2(x_trans + self.dropout(cross_attn_output))
            ffn_output = self.ffn(x_trans)
            x_trans = self.norm3(x_trans + self.dropout(ffn_output))
        else:
            ffn_output = self.ffn(x_trans)
            x_trans = self.norm2(x_trans + self.dropout(ffn_output))
        
        return x_trans.transpose(0, 1)

class SimpleTransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_enc_layers=12, num_dec_layers=12,
                 num_heads=8, d_ff=2048, max_seq_len=500, dropout=0.1):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_seq_len)
        # Encoder stack
        self.encoder_layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_enc_layers)]
        )
        # Decoder stack with cross-attention
        self.decoder_layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, dropout, is_decoder=True) for _ in range(num_dec_layers)]
        )
        self.fc_out = nn.Linear(d_model, vocab_size)
    
    def encode(self, src, src_mask=None):
        src_emb = self.token_embed(src)
        src_emb = self.pos_enc(src_emb)
        for layer in self.encoder_layers:
            src_emb = layer(src_emb, attn_mask=src_mask)
        return src_emb

    def decode(self, tgt, enc_output, tgt_mask=None, cross_attn_mask=None):
        tgt_emb = self.token_embed(tgt)
        tgt_emb = self.pos_enc(tgt_emb)
        for layer in self.decoder_layers:
            tgt_emb = layer(tgt_emb, enc_output=enc_output, attn_mask=tgt_mask, cross_attn_mask=cross_attn_mask)
        return tgt_emb

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, cross_attn_mask=None):
        enc_output = self.encode(src, src_mask=src_mask)
        dec_output = self.decode(tgt, enc_output, tgt_mask=tgt_mask, cross_attn_mask=cross_attn_mask)
        logits = self.fc_out(dec_output)
        return logits

    def generate(self, input_ids, max_length, num_beams, early_stopping, tokenizer):
        # Implement beam search or other decoding strategies here
        # For simplicity, this is a placeholder implementation
        output = input_ids
        for _ in range(max_length):
            logits = self.forward(output, output)
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
            output = torch.cat((output, next_token_id), dim=1)
            if early_stopping and next_token_id.item() == tokenizer.eos_token_id:
                break
        return output

if __name__ == "__main__":
    vocab_size = 5000
    model = SimpleTransformerLM(vocab_size)
    src = torch.randint(0, vocab_size, (16, 50))
    tgt = torch.randint(0, vocab_size, (16, 50))
    out = model(src, tgt)
    print(out.shape)  # Expected: (16, 50, vocab_size)
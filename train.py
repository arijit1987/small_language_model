import torch
import torch.nn as nn
import torch.optim as optim
from model import SimpleTransformerLM
from data import load_custom_dataset, tokenize_dataset, get_dataloaders
from transformers import GPT2Tokenizer

def train_model(model, train_dataloader, val_dataloader, tokenizer, epochs=20, lr=1e-4, device="cpu"):
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_dataloader:
            batch = batch.to(device)
            # Ensure the input tensors are of type Long
            batch = batch.long()
            # Split batch into source (encoder input) and target (decoder input)
            src = batch[:, :-1]
            tgt = batch[:, 1:]
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            optimizer.zero_grad()
            # Using src as encoder input and tgt_input as decoder input
            logits = model(src, tgt_input)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.4f}")
        scheduler.step()
    return model

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Load and tokenize dataset
    ds = load_custom_dataset()
    tokenized_ds = tokenize_dataset(ds, tokenizer, max_length=50)
    train_loader, val_loader = get_dataloaders(tokenized_ds, tokenizer, batch_size=16)
    vocab_size = tokenizer.vocab_size
    model = SimpleTransformerLM(
        vocab_size,
        d_model=512,
        num_enc_layers=6,
        num_dec_layers=6,
        num_heads=8,
        d_ff=2048,
        max_seq_len=50,
        dropout=0.1
    )
    trained_model = train_model(model, train_loader, val_loader, tokenizer, epochs=20, lr=1e-4, device=device)
    torch.save(trained_model.state_dict(), "small_transformer_model.pth")

if __name__ == "__main__":
    main()
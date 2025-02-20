from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import GPT2Tokenizer
from model import SimpleTransformerLM
import uvicorn

app = FastAPI()

# Define the request body model
class QueryRequest(BaseModel):
    query: str

# Load the tokenizer and initialize the model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
vocab_size = tokenizer.vocab_size
model = SimpleTransformerLM(
    vocab_size,
    d_model=512,
    num_enc_layers=6,
    num_dec_layers=6,
    num_heads=8,
    d_ff=2048,
    max_seq_len=50,  # Ensure this matches the value used during training
    dropout=0.1
)
model.load_state_dict(torch.load("small_transformer_model.pth", map_location=torch.device("cpu")))
model.eval()

@app.post("/chat")
async def chat(request: QueryRequest):
    # Tokenize the query
    input_ids = tokenizer.encode(request.query, return_tensors="pt")
    # Ensure input sequence length does not exceed max_seq_len
    input_ids = input_ids[:, :50]
    # Generate response using beam search
    max_length = 50
    num_beams = 5
    with torch.no_grad():
        output = model.generate(input_ids, max_length=max_length, num_beams=num_beams, early_stopping=True, tokenizer=tokenizer)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"response": response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
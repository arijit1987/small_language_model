Token Usage:
GitHub Tokens: 3335
LLM Input Tokens: 3345
LLM Output Tokens: 1316
Total Tokens: 7996

FileTree:
.gitignore
api.py
client.py
data.py
main.py
model.py
requirements.txt
train.py

Analysis:
```markdown
# Medical Chatbot with Simple Transformer

This project implements a simple medical chatbot using a Transformer model. It's built with PyTorch, Transformers library, FastAPI for API deployment, and leverages a medical conversation dataset.

## Project Structure

```
.
├── .gitignore          # Specifies intentionally untracked files that Git should ignore
├── api.py              # FastAPI application for serving the chatbot
├── client.py           # Client script to interact with the API
├── data.py             # Data loading and preprocessing utilities
├── main.py             # Entry point for training or running the API
├── model.py            # Definition of the Transformer model
├── README.md           # This file
├── requirements.txt    # List of Python dependencies
├── small_transformer_model.pth # Trained model weights (will be created after training)
├── train.py            # Training script for the Transformer model
└── __pycache__/        # Python cache directory (ignored by Git)
```

## Setup and Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd medical-chatbot
    ```

2.  **Create and activate a virtual environment (recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate  # On Windows
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Training the Model

To train the Transformer model, run the following command:

```bash
python main.py --mode train
```

This will:

*   Load the medical conversation dataset from Hugging Face Datasets (`DSWF/medical_chatbot`).
*   Tokenize the dataset using the GPT-2 tokenizer.
*   Train the `SimpleTransformerLM` model.
*   Save the trained model weights to `small_transformer_model.pth`.

**Note:** Training can take a significant amount of time, depending on your hardware.  The default training configuration uses 20 epochs.  You can adjust this in `train.py`.

### 2. Running the API

To start the FastAPI server and expose the chatbot API, run:

```bash
python main.py --mode api
```

This will start the API server at `http://0.0.0.0:8000`.  The `--reload` flag in `main.py` enables automatic reloading of the server when code changes are detected, which is useful for development.

### 3. Interacting with the API

Use the `client.py` script to send queries to the chatbot API:

```bash
python client.py
```

This script sends a sample medical query to the API and prints the response.  You can modify the `payload` in `client.py` to test different queries.

The API endpoint is `/chat`, and it expects a JSON payload with a `query` field.

Example:

```json
{
  "query": "Hi doctor, I have a headache and a fever."
}
```

The API will return a JSON response with a `response` field containing the chatbot's answer.

Example:

```json
{
  "response": "Based on your symptoms, it sounds like you may have a cold or the flu. I recommend getting some rest and drinking plenty of fluids."
}
```

## Model Details

The `model.py` file defines the `SimpleTransformerLM` class, which implements a basic Transformer model for language modeling.  It includes:

*   **Positional Encoding:**  Adds positional information to the input embeddings.
*   **Transformer Blocks:**  Implement the encoder and decoder layers with multi-head attention and feed-forward networks.
*   **Encoder and Decoder Stacks:**  The model uses a stack of encoder and decoder layers.
*   **Generate Function:**  A simple implementation of a generation strategy.  Currently, it's a placeholder and can be improved with beam search or other decoding methods.

## Data Details

The `data.py` file handles loading and preprocessing the medical conversation dataset. It uses the `DSWF/medical_chatbot` dataset from Hugging Face Datasets. The dataset is tokenized using the GPT-2 tokenizer.  The `collate_fn` function is used to pad the sequences in each batch to the same length.

## Important Considerations

*   **Model Size:** The `SimpleTransformerLM` model is designed to be relatively small for demonstration purposes.  For better performance, you can increase the model size (e.g., `d_model`, `num_enc_layers`, `num_dec_layers`, `d_ff`).
*   **Training Data:** The performance of the chatbot depends heavily on the quality and quantity of the training data.  Consider using a larger and more diverse medical conversation dataset for better results.
*   **Decoding Strategy:** The `generate` function in `model.py` uses a very basic decoding strategy.  Implement beam search or other advanced decoding techniques for improved text generation.
*   **Ethical Considerations:**  Medical chatbots should be used with caution and should not be considered a substitute for professional medical advice.  It's important to ensure that the chatbot provides accurate and safe information.  This project is for educational purposes only and should not be used for real-world medical applications without careful consideration of ethical and safety implications.
*   **Max Sequence Length:** The `max_seq_len` parameter is set to 50.  This limits the length of the input and output sequences.  Ensure this value is appropriate for your use case.  In `api.py`, the input is truncated to this length as well.
*   **Device:** The training script automatically detects and uses MPS (Apple Silicon) if available, otherwise it defaults to CPU. You can explicitly set the device in `train.py`.

## Contributing

Contributions are welcome!  Please feel free to submit pull requests with bug fixes, improvements, or new features.
```

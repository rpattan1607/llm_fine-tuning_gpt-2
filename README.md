# Fine-Tuning and LLMs from Scratch
This repository contains code for building LLMs from scratch and also finetuning of Pre-Trained Model GPT2 for Classification and Instruction Tuning.

## Key features and functionalities:

- **Tokenizer Implementation**: Custom tokenizer to preprocess and tokenize input text data for transformer-based models.

- **Embeddings Model**: Build embedding layers to map input tokens into high-dimensional vector representations.

- **Data Loaders**: Efficiently manage and prepare datasets for training and evaluation using custom data loaders.

- **Multi-Head Self-Attention Mechanism**: Implement the multi-head self-attention mechanism to capture relationships between tokens in the input sequence.

- **Transformer Block Construction**: Assemble a complete transformer block by combining self-attention, feed-forward layers, and residual connections.

- **Loading Pretrained Models**: Load and utilize pretrained transformer models for initialization and transfer learning.

- **Model Finetuning**: Finetune pretrained transformer models on custom datasets to improve task-specific performance.

- **Model Evaluation**: Evaluate model performance using appropriate metrics and generate insights on the results.

- **Flexible and Modular Design**: Designed with modularity to allow easy integration and experimentation with different components.

### GPT-2 Model:

The GPT-2 (Generative Pre-trained Transformer 2) model is a state-of-the-art transformer-based language model developed by OpenAI. It is designed for natural language processing tasks such as text generation, translation, summarization, and more. GPT-2 uses a decoder-only architecture with multi-head self-attention mechanisms and is trained on a diverse dataset to predict the next word in a sentence, making it adept at generating coherent and contextually relevant text. Its ability to generate human-like text has made it a popular choice for numerous applications in AI research and development.

![image](https://github.com/user-attachments/assets/233c8346-1788-4f7e-9ac2-be3639b63515)
**Source**: [GPT-lite by Bruno Maga](https://brunomaga.github.io/GPT-lite)

## Requirements 

To execute the project, ensure you have the following dependencies installed:

- `torch` (for `torch.utils.data.DataLoader`): PyTorch library for building and training models.
- `tiktoken`: A library for efficient tokenization, used with transformer models.
- `transformers`: For loading and using the `GPT2LMHeadModel` and `GPT2Tokenizer`.
- `json`: Built-in Python library for working with JSON data.
- `urllib.request`: Built-in Python module for making HTTP requests.
- `requests`: Library for handling HTTP requests and APIs.
- `zipfile`: Built-in Python module for extracting and handling zip files.
- `pandas`: Library for data manipulation and analysis.

## References 

- https://github.com/rasbt/LLMs-from-scratch/tree/main
- https://brunomaga.github.io/GPT-lite
- https://arxiv.org/abs/1706.03762
- https://paperswithcode.com/method/gpt-2
  

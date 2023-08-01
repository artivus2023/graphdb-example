import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List

# Average pooling
def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

# List of sentences embedding
def create_embeddings(input_texts, model_name='intfloat/e5-large-v2'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Tokenize the input texts
    batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')

    outputs = model(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    scores = (embeddings[:2] @ embeddings[2:].T) * 100
    return scores.tolist()

# Single sentence embedding
def create_embedding(sentence, model_name='intfloat/e5-large-v2'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Tokenize the sentence
    inputs = tokenizer(sentence, return_tensors='pt')

    outputs = model(**inputs)
    embeddings = average_pool(outputs.last_hidden_state, inputs['attention_mask'])

    # Normalize the embedding
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    return embeddings.squeeze().tolist()


def compare_embeddings(embeddings1: List[float], embeddings2: List[List[float]]):
    """
    Compares two sets of embeddings by calculating the cosine similarity between each pair of embeddings.

    Parameters:
    embeddings1 (List[float]): The first set of embeddings. These will be compared to each embedding in embeddings2.
    embeddings2 (List[List[float]]): The second set of embeddings.

    Returns:
    List[Tuple[int, float]]: A list of tuples, where each tuple contains the index and cosine similarity of one of the top 10 most similar embeddings in embeddings2 to embeddings1.
    """
    cos_sims = []
    for embedding in embeddings2:
        cos_sim = np.dot(embeddings1, embedding) / (np.linalg.norm(embeddings1) * np.linalg.norm(embedding))
        cos_sims.append(cos_sim)

    # Convert to numpy array for easier manipulation
    cos_sims = np.array(cos_sims)

    # Get the indices of the top 10 cosine similarities
    top_10_indices = cos_sims.argsort()[-10:][::-1]

    # Get the top 10 cosine similarities
    top_10_values = cos_sims[top_10_indices]

    return list(zip(top_10_indices, top_10_values))


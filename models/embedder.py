from typing import List, Optional

import torch
# Please run `pip install -U sentence-transformers`
from sentence_transformers import SentenceTransformer
from torch import Tensor


class SBTextEmbedding:
    def __init__(self, device: Optional[torch.device] = None):
        self.model = SentenceTransformer(
            'sentence-transformers/all-MiniLM-L12-v2',
            device=str(device),
        )

    def __call__(self, sentences: List[str]) -> Tensor:
        return torch.from_numpy(self.model.encode(sentences))
    

class GloveTextEmbedding:
    def __init__(self, device: Optional[torch.device] = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = SentenceTransformer(
            "sentence-transformers/average_word_embeddings_glove.6B.300d",
            device=str(device),
        )

    def __call__(self, sentences: List[str]) -> Tensor:
        return torch.from_numpy(self.model.encode(sentences))
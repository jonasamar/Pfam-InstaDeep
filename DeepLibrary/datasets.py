
import torch

def encode(X, tokenizer, max_length = 1e4):
    """
    Encode a list of input sequences using the provided tokenizer.

    Parameters:
    - X (list): List of input sequences.
    - tokenizer: Tokenizer object for encoding sequences.
    - max_length (int): Maximum length of the encoded sequences.

    Returns:
    - result (list): List of encoded sequences.
    """
    result = []
    for x in X:
        ids = tokenizer.encode(x).ids
        # Truncate to the specified max_length
        ids = ids[:max_length]
        result.append(ids)
    return result
    
def batch_iterator(dataset, batch_size = 10000):
    """
    Generate batches from a dataset.

    Parameters:
    - dataset (list): List of data samples.

    Yields:
    - batch (list): Batch of data samples.
    """
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]


class Dataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset class for handling encoded sequences and labels.

    Parameters:
    - encodings (list): List of encoded sequences.
    - labels: List of corresponding labels.
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels 

    def __getitem__(self, idx):
        return self.encodings[idx], self.labels[idx]

    def __len__(self):
        return len(self.encodings)
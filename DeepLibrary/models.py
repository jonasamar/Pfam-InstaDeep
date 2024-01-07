import torch.nn as nn
import torch
from transformers import BertModel, BertConfig, BertTokenizerFast
import torch.nn.functional as F

class BioBERTModel(nn.Module):
    """
    BioBERTModel is a PyTorch implementation of a transformer-based model for
    processing biological sequence data. It utilizes the BioBERT architecture,
    which is pre-trained on domain-specific corpora to capture meaningful
    patterns in biological sequences. This model is designed for classification
    or regression tasks on biological data.

    Parameters:
    - hidden_size (int): Size of the hidden layers in the transformer.
    - num_layers (int): Number of hidden layers in the transformer.
    - num_attention_heads (int): Number of attention heads in the transformer.
    - num_classes (int): Number of output classes for the classification task.
    """
    
    def __init__(self, hidden_size, num_layers, num_attention_heads, num_classes):
        super(BioBERTModel, self).__init__()
        configuration = BertConfig(hidden_size=hidden_size, num_hidden_layers=num_layers, num_attention_heads=num_attention_heads)
        self.transformer = BertModel(configuration)
        
        # additional layers for the classification / regression task
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
       )

    def forward(self, ids, mask=None, token_type_ids=None):
        sequence_output, pooled_output = self.transformer(
           ids, 
           attention_mask=mask,
           token_type_ids=token_type_ids,
           return_dict=False
        )

        sequence_output = torch.mean(sequence_output, dim=1)
        result = self.head(sequence_output)
        
        return result



class DynamicConvolutionalModel(nn.Module):
    def __init__(self, input_size, conv_configs, num_classes):
        """
        Initialize the DynamicConvolutionalModel.

        Parameters:
            - input_size (int): Size of the input features.
            - conv_configs (list): List of tuples, each containing (filter_size, num_filters).
            - num_classes (int): Number of classes in the classification task.
        """
        super(DynamicConvolutionalModel, self).__init__()
        
        # Convolutional Layers
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(in_channels=input_size, out_channels=num_filters, kernel_size=filter_size)
            for filter_size, num_filters in conv_configs
        ])

        # Fully Connected Layer
        self.fc = nn.Linear(sum(num_filters for _, num_filters in conv_configs), num_classes)

    def forward(self, x):
        """
        Forward pass of the model.

        Parameters:
            - x (torch.Tensor): Input tensor representing sequences.

        Returns:
            - torch.Tensor: Output tensor representing class probabilities.
        """
        # Permute dimensions for Conv1D
        x = x.permute(0, 2, 1)

        # Convolutional Layers with ReLU activation
        x = [F.relu(conv(x)) for conv in self.conv_layers]

        # Max pooling over time
        x = [F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2) for conv_out in x]

        # Concatenate the results
        x = torch.cat(x, dim=1)

        # Fully Connected Layer
        x = self.fc(x)

        return F.log_softmax(x, dim=1)

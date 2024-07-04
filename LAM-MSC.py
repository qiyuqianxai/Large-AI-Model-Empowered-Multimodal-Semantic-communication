import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
import random
import numpy as np
import json
from channel_nets import channel_net
import os
import SCwithCGE
import MMA
import LKB

class params():
    # Configuration parameters for training and testing
    checkpoint_path = "checkpoints"  # Path to save model checkpoints
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available
    dataset = r"E:\datasets\VOC2012_img2text"  # Path to dataset
    log_path = "logs"  # Path to save logs
    epoch = 100  # Number of training epochs
    lr = 1e-3  # Learning rate
    batchsize = 16  # Batch size for training
    snr = 15  # Signal-to-noise ratio
    weight_delay = 1e-5  # Weight decay for optimizer
    sim_th = 0.6  # Similarity threshold for evaluation
    emb_dim = 768  # Embedding dimension
    n_heads = 8  # Number of attention heads in the transformer
    hidden_dim = 1024  # Hidden dimension in the transformer
    num_layers = 2  # Number of layers in the transformer
    use_CGE = False  # Whether to use channel gain estimation (CGE)
    max_length = 30  # Maximum sequence length for tokenization

def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    img_path = "imgs"
    # Modal transformation based on MMA
    texts = MMA.img2text(img_path)
    # Semantic Extraction Based on LKB
    userInfo = {"name": "Mike", "interests": "running", "language": "English", "identify": "student", "gender": "male"}
    personalized_semantics = []
    for input_text in texts:
        personalized_semantic = LKB.personalized_semantics(userInfo, input_text)
        personalized_semantics.append(personalized_semantic)
    # Data Transmission Based on CGE Assisted-SC
    rec_texts = SCwithCGE.data_transmission(personalized_semantics)
    # Semantic Recovery Based on LKB
    userInfo = {"name": "Jane", "interests": "shopping", "language": "English", "identify": "student","gender":"female"}
    personalized_semantics = []
    for input_text in rec_texts:
        personalized_semantic = LKB.personalized_semantics(userInfo, input_text)
        personalized_semantics.append(personalized_semantic)
    # Modal recovery based on MMA
    MMA.text2img(personalized_semantics)
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
import random
import numpy as np
import json
from channel_nets import channel_net, sample_batch, MutualInfoSystem
import os
import SCwithCGE
import MMA
import LKB

class params():
    checkpoint_path = "checkpoints"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = r"E:\datasets\VOC2012_img2text"
    log_path = "logs"
    epoch = 100
    lr = 1e-3
    batchsize = 16
    snr = 15
    weight_delay = 1e-5
    sim_th = 0.6
    emb_dim = 768
    n_heads = 8
    hidden_dim = 1024
    num_layers = 2
    use_CGE = False
    max_length = 30

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
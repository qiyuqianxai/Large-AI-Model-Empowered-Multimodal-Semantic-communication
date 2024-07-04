import copy
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
import random
import numpy as np
import json
from channel_nets import channel_net
import os

class params():
    # Configuration parameters for training and testing
    checkpoint_path = "checkpoints"  # Path to save model checkpoints
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available
    dataset = r"E:\datasets\VOC2012_img2text"  # Path to dataset
    log_path = "logs"  # Path to save logs
    epoch = 20  # Number of training epochs
    lr = 1e-3  # Learning rate
    batchsize = 16  # Batch size for training
    snr = 15  # Signal-to-noise ratio
    weight_delay = 1e-5  # Weight decay for optimizer
    sim_th = 0.6  # Similarity threshold for evaluation
    emb_dim = 768  # Embedding dimension
    n_heads = 8  # Number of attention heads in the transformer
    hidden_dim = 1024  # Hidden dimension in the transformer
    num_layers = 2  # Number of layers in the transformer
    use_CGE = True  # Whether to use channel gain estimation (CGE)
    max_length = 30  # Maximum sequence length for tokenization


def same_seeds(seed):
    # Set random seeds for reproducibility
    random.seed(seed)  # Python built-in random module
    np.random.seed(seed)  # Numpy
    torch.manual_seed(seed)  # Torch
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # Torch CUDA
        torch.cuda.manual_seed_all(seed)  # All Torch CUDA devices
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class TextSCNet(nn.Module):
    def __init__(self, emb_dim, n_heads, hidden_dim, num_layers):
        super(TextSCNet, self).__init__()
        # Define the encoder as BERT
        self.encoder = BertModel.from_pretrained('bert-base-uncased')
        # Define the decoder as a Transformer decoder
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(emb_dim, n_heads, hidden_dim), num_layers)
        # Fully connected layer for output
        self.fc = nn.Linear(emb_dim, self.encoder.config.vocab_size)
        # Channel model
        self.channel_model = channel_net(in_dims=23040, snr=arg.snr, CGE=arg.use_CGE)
        # Embedding layer
        self.embedding = nn.Embedding(self.encoder.config.vocab_size, emb_dim)

    def forward(self, src_input_ids, src_attention_mask, trg_input_ids):
        # Encode the source inputs
        s_code = self.encoder(src_input_ids, attention_mask=src_attention_mask).last_hidden_state
        b_s, w_s, f_s = s_code.shape
        s_code = s_code.view(b_s, -1)
        # Pass through the channel model
        ch_code, ch_code_, s_code_d = self.channel_model(s_code)
        # Decode the target inputs
        trg_emb = self.embedding(trg_input_ids)
        s_code_ = s_code_d.view(b_s, w_s, f_s)
        decoded = self.decoder(trg_emb, s_code_)
        decoded_output = self.fc(decoded)
        return ch_code, ch_code_, s_code, s_code_d, decoded_output

def SC_train(model, tokenizer, training_texts, arg):
    model.to(arg.device)
    # Define optimizer
    criterion = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    weights_path = f"{arg.checkpoint_path}/TextSC_snr{arg.snr}.pth"
    raw_text = []
    rec_text = []
    for epoch in range(arg.epoch):
        random.shuffle(training_texts)
        model.train()
        for i in range(0, len(training_texts), arg.batchsize):
            if i + arg.batchsize < len(training_texts):
                b_text = training_texts[i:i + arg.batchsize]
            else:
                break
            raw_text += b_text
            optimizer.zero_grad()
            # Tokenize input text
            encoded_dict = tokenizer.batch_encode_plus(
                b_text, add_special_tokens=True, max_length=arg.max_length,
                pad_to_max_length=True, return_attention_mask=True, return_tensors='pt'
            )
            input_ids = encoded_dict['input_ids'].to(arg.device)
            encoded_dict_d = tokenizer.batch_encode_plus(
                b_text, add_special_tokens=True, max_length=arg.max_length,
                pad_to_max_length=True, return_attention_mask=True, return_tensors='pt'
            )
            target_ids = encoded_dict_d['input_ids'].to(arg.device)

            src_input_ids = input_ids.clone()
            trg_input_ids = target_ids.clone()
            src_attention_mask = (src_input_ids != tokenizer.pad_token_id).float().to(arg.device)
            # Forward pass through the model
            ch_code, ch_code_, s_code, s_code_, output = model(src_input_ids, src_attention_mask, trg_input_ids)

            # Compute losses
            loss_ch = mse(s_code, s_code_)
            loss_SC = criterion(output.view(-1, model.encoder.config.vocab_size), target_ids.contiguous().view(-1))
            loss = loss_SC + loss_ch
            loss.backward()
            optimizer.step()

            # Recover the text
            for i, o in enumerate(output):
                predicted_indices = torch.argmax(o.view(-1, model.encoder.config.vocab_size), dim=1).cpu().numpy()
                predicted_sentence = tokenizer.decode(predicted_indices, skip_special_tokens=True)
                print("src:", b_text[i], '\nrec:', predicted_sentence)
                rec_text.append(predicted_sentence)
            with open(os.path.join(arg.log_path, f"t2t_snr{arg.snr}_res.json"), "w", encoding="utf-8") as f:
                f.write(json.dumps({"raw_text": raw_text, "rec_text": rec_text}, indent=4, ensure_ascii=False))
            print(f"epoch {epoch}, loss: {loss.item()}")
        torch.save(model.state_dict(), weights_path)


@torch.no_grad()
def data_transmission(input_text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    SC_model = TextSCNet(arg.emb_dim, arg.n_heads, arg.hidden_dim, arg.num_layers).to(arg.device)
    weight = torch.load(f"{arg.checkpoint_path}/TextSC_snr{arg.snr}.pth", map_location="cpu")
    SC_model.load_state_dict(weight)
    SC_model.eval()
    encoded_dict = tokenizer.batch_encode_plus(
        input_text, add_special_tokens=True, max_length=arg.max_length,
        pad_to_max_length=True, return_attention_mask=True, return_tensors='pt'
    )
    input_ids = encoded_dict['input_ids'].to(arg.device)
    src_input_ids = input_ids.clone()
    src_attention_mask = (src_input_ids != tokenizer.pad_token_id).float().to(arg.device)
    encoded_dict_d = tokenizer.batch_encode_plus(
        input_text, add_special_tokens=True, max_length=arg.max_length,
        pad_to_max_length=True, return_attention_mask=True, return_tensors='pt'
    )
    trg_input_ids = encoded_dict_d['input_ids'].to(arg.device)
    ch_code, ch_code_, s_code, s_code_, output = SC_model(src_input_ids, src_attention_mask, trg_input_ids)
    rec_text = []
    for o in output:
        predicted_indices = torch.argmax(o.view(-1, SC_model.encoder.config.vocab_size), dim=1).cpu().numpy()
        predicted_sentence = tokenizer.decode(predicted_indices, skip_special_tokens=True)
        rec_text.append(predicted_sentence)
    print(rec_text)
    return rec_text


arg = params()
if __name__ == '__main__':
    same_seeds(1024)
    train_data = []
    for text in os.listdir(arg.dataset):
        if text.endswith(".json"):
            text_path = os.path.join(arg.dataset, text)
            with open(text_path, "r", encoding="utf-8") as f:
                content = json.load(f)
                content = [val.replace("<unk>", "") for val in content]
            train_data += content
    print(arg.device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    SC_model = TextSCNet(arg.emb_dim, arg.n_heads, arg.hidden_dim, arg.num_layers).to(arg.device)
    SC_train(SC_model, tokenizer, train_data, arg)
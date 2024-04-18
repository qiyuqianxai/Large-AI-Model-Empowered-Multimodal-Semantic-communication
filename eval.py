import copy
import json
from transformers import MarianMTModel, MarianTokenizer
import torch
from torch import nn
import os
import warnings
import torchvision.datasets as dset
from PIL import Image
warnings.filterwarnings("ignore")
from base_nets import base_net
from channel_nets import channel_net
import time
import numpy as np
import torchvision
import random

torch.cuda.set_device(0)
class params():
    checkpoint_path = "checkpoints"
    device = "cuda"
    dataset = r"E:\pengyubo\datasets\UCF100"
    log_path = "logs"
    epoch = 5
    lr = 1e-3
    batchsize = 8
    snr = 25
    weight_delay = 1e-5
    sim_th = 0.6

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

@torch.no_grad()
def test(tr_model, channel_net, tokenizer, training_texts, arg):
    tr_model = tr_model.to(arg.device)
    tr_model.eval()
    channel_model = channel_net.to(arg.device)
    weights_path = f"{arg.checkpoint_path}/t2t_ch_snr{arg.snr}.pth"
    weight = torch.load(weights_path, map_location="cpu")
    channel_model.load_state_dict(weight)
    channel_model.eval()
    raw_text = []
    rec_text = []
    random.shuffle(training_texts)
    for i in range(0, len(training_texts), arg.batchsize):
        if i + arg.batchsize < len(training_texts):
            b_text = training_texts[i:i + arg.batchsize]
        else:
            b_text = training_texts[i:]
        # Tokenize the input text
        input_ids = tokenizer.batch_encode_plus(b_text, return_tensors="pt", padding=True, max_length=512)[
            "input_ids"].to(arg.device)
        # input_ids = tokenizer.encode(b_text, return_tensors="pt").to(arg.device)
        # Encode the input text
        encoder_outputs = tr_model.get_encoder()(input_ids)
        model_inputs = {
            "input_ids": None,
            "encoder_outputs": encoder_outputs,
            "past_key_values": None,
        }
        decoded_ids = tr_model.generate(**model_inputs)
        # Decode the generated ids to text
        translated_texts = [tokenizer.decode(t, skip_special_tokens=True) for t in decoded_ids]
        raw_text += translated_texts
        # print(translated_texts)

        shape = encoder_outputs[0].shape
        encoder_outputs_temp = encoder_outputs[0].view(-1, 512)
        encoder_outputs_with_noise = channel_model(encoder_outputs_temp)
        encoder_outputs[0].data = encoder_outputs_with_noise.view(shape).data
        model_inputs = {
            "input_ids": None,
            "encoder_outputs": encoder_outputs,
            "past_key_values": None,
        }
        decoded_ids = tr_model.generate(**model_inputs)
        # Decode the generated ids to text
        translated_texts = [tokenizer.decode(t, skip_special_tokens=True) for t in decoded_ids]
        rec_text += translated_texts
        # print(translated_texts)

    with open(os.path.join(arg.log_path, f"t2t_snr{arg.snr}_eval_res.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps({"raw_text": raw_text, "rec_text": rec_text}, indent=4, ensure_ascii=False))
    # evaluate(raw_text, rec_text,tokenizer, tr_model, arg)

@torch.no_grad()
def evaluate(src_txts, tar_txts, arg):
    from transformers import BertTokenizer, BertModel
    from sklearn.metrics.pairwise import cosine_similarity

    # Load the BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model = BertModel.from_pretrained('bert-base-chinese')
    acc = 0
    cos_sims = []
    for src_txt, tar_txt in zip(src_txts,tar_txts):
        # Tokenize and process each sentence individually
        encoded_sentence1 = tokenizer.encode_plus(src_txt, add_special_tokens=True, max_length=64,
                                                  truncation=True, return_tensors='pt', padding='max_length')
        encoded_sentence2 = tokenizer.encode_plus(tar_txt, add_special_tokens=True, max_length=64,
                                                  truncation=True, return_tensors='pt', padding='max_length')

        # Obtain the BERT embeddings for each sentence

        model_output1 = model(encoded_sentence1['input_ids'], encoded_sentence1['attention_mask'])
        embeddings1 = model_output1.last_hidden_state[:, 0, :]

        model_output2 = model(encoded_sentence2['input_ids'], encoded_sentence2['attention_mask'])
        embeddings2 = model_output2.last_hidden_state[:, 0, :]

        # Calculate the similarity using cosine similarity
        similarity = cosine_similarity(embeddings1, embeddings2)[0][0]
        # print(f"Cosine similarity score: {similarity}")
        cos_sims.append(similarity)
        if similarity > arg.sim_th:
            acc+=1
    # print("SNR:",arg.snr," accuracy:",acc/len(src_txts))
    cos_sims = np.array(cos_sims)
    accs = []
    ths = []
    for i in range(60,90,2):
        th = i/100
        ths.append(th)
        acc = cos_sims[cos_sims > th]
        acc = len(acc)/len(cos_sims)
        accs.append(acc)
    print("ths:",ths)
    print("accs:",accs)

if __name__ == '__main__':
    same_seeds(1024)
    arg = params()
    training_texts = []
    for text in os.listdir(arg.dataset):
        if text.endswith(".json"):
            text_path = os.path.join(arg.dataset,text)
            with open(text_path,"r",encoding="utf-8")as f:
                content = json.load(f)
                content = [val.replace("<unk>","") for val in content]
            training_texts+=content
    print(len(training_texts))
    model_name = "Helsinki-NLP/opus-mt-en-zh"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    tr_model = MarianMTModel.from_pretrained(model_name)


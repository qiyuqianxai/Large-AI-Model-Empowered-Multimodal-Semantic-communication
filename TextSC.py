import copy
import json
from transformers import MarianMTModel, MarianTokenizer
from transformers import BertTokenizer, BertModel
import torch
from torch import nn
import os
import warnings
import torchvision.datasets as dset
from PIL import Image
warnings.filterwarnings("ignore")
from base_nets import base_net
from channel_nets import channel_net, multipath_generator
import time
import numpy as np
import torchvision
import random
torch.cuda.set_device(0)
class params():
    checkpoint_path = "checkpoints"
    device = "cuda"
    dataset = r"E:\datasets\VOC2012_img2text"
    log_path = "logs"
    epoch = 100
    lr = 1e-3
    batchsize = 16
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

def SC_train(tr_model,channel_net,tokenizer,training_texts, arg):
    tr_model = tr_model.to(arg.device)
    channel_model = channel_net.to(arg.device)
    # define optimizer
    optimizer = torch.optim.Adam(channel_model.parameters(), lr=arg.lr,
                                             weight_decay=arg.weight_delay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                      factor=0.1, patience=300,
                                                                      verbose=True, threshold=0.0001,
                                                                      threshold_mode='rel',
                                                                      cooldown=0, min_lr=0, eps=1e-08)
    weights_path = f"{arg.checkpoint_path}/t2t_ch_snr{arg.snr}.pth"
    mse = nn.MSELoss()
    raw_text = []
    rec_text = []
    for epoch in range(arg.epoch):
        random.shuffle(training_texts)
        for i in range(0,len(training_texts),arg.batchsize):
            if i+arg.batchsize < len(training_texts):
                b_text = training_texts[i:i+arg.batchsize]
            else:
                b_text = training_texts[i:]
            raw_text += b_text
            print(f"input text:{b_text}")
            # Tokenize the input text
            input_ids = tokenizer.batch_encode_plus(b_text,
                add_special_tokens=True,  # Add special tokens, such as [CLS] and [SEP]
                max_length=30,  # Set the maximum length, truncate if exceeded
                pad_to_max_length=True,  # Pad to the maximum length
                return_attention_mask=True,  # Return attention masks
                return_tensors='pt'  # Return tensor type, here is PyTorch
                )["input_ids"].to(arg.device)
            with torch.no_grad():
                encoder_outputs = tr_model(input_ids)[0]
            shape = encoder_outputs[0].shape
            encoder_outputs_temp = encoder_outputs[0].view(-1,512)
            h_I, h_Q = multipath_generator(encoder_outputs_temp.shape[0])
            ch_code,ch_code_with_n,x = channel_model(encoder_outputs_temp,h_I,h_Q)
            # compute channel loss
            loss_MI = mse(ch_code,ch_code_with_n)# use mse(ch_code,ch_code_with_n) for simplicity
            loss_SC = mse(x,encoder_outputs_temp)
            loss = loss_MI + loss_SC
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            encoder_outputs[0].data = x.view(shape).data
            # Decode the input using the encoder outputs and decoder input ids
            model_inputs = {
                "input_ids": None,
                "encoder_outputs": encoder_outputs,
                "past_key_values": None,
            }
            with torch.no_grad():
                decoded_ids = tr_model(**model_inputs)
            # Decode the generated ids to text
            translated_texts = [tokenizer.decode(t, skip_special_tokens=True) for t in decoded_ids]
            print(translated_texts)
            rec_text += translated_texts
            with open(os.path.join(arg.log_path,f"t2t_snr{arg.snr}_res.json"),"w",encoding="utf-8")as f:
                f.write(json.dumps({"raw_text":raw_text,"rec_text":rec_text},indent=4,ensure_ascii=False))
            print(f"epoch {epoch}, loss: {loss.item()}")
        torch.save(channel_model.state_dict(), weights_path)

@torch.no_grad()
def SC_test(tr_model, channel_net, tokenizer, training_texts, arg):
    tr_model = tr_model.to(arg.device)
    tr_model.eval()
    channel_model = channel_net.to(arg.device)
    # weights_path = f"{arg.checkpoint_path}/t2t_ch_snr{arg.snr}.pth"
    # weight = torch.load(weights_path, map_location="cpu")
    # channel_model.load_state_dict(weight)
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
        input_ids = tokenizer.batch_encode_plus(b_text,
                                                add_special_tokens=True,  # Add special tokens, such as [CLS] and [SEP]
                                                max_length=30,  # Set the maximum length, truncate if exceeded
                                                pad_to_max_length=True,  # Pad to the maximum length
                                                return_attention_mask=True,  # Return attention masks
                                                return_tensors='pt'  # Return tensor type, here is PyTorch
                                                )["input_ids"].to(arg.device)

        encoder_outputs = tr_model.get_encoder()(input_ids)
        shape = encoder_outputs[0].shape
        encoder_outputs_temp = encoder_outputs[0].view(-1, 512)
        h_I, h_Q = multipath_generator(encoder_outputs_temp.shape[0])
        ch_code, ch_code_with_n, x = channel_model(encoder_outputs_temp, h_I, h_Q)
        encoder_outputs[0].data = x.view(shape).data
        # Decode the input using the encoder outputs and decoder input ids
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
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    acc = 0
    cos_sims = []
    for src_txt, tar_txt in zip(src_txts, tar_txts):
        # Tokenize and process each sentence individually
        encoded_sentence1 = tokenizer.encode_plus(src_txt,
        add_special_tokens=True,  # Add special tokens, such as [CLS] and [SEP]
        max_length=30,  # Set the maximum length, truncate if exceeded
        pad_to_max_length=True,  # Pad to the maximum length
        return_attention_mask=True,  # Return attention masks
        return_tensors='pt'  # Return tensor type, here is PyTorch
        )
        encoded_sentence2 = tokenizer.encode_plus(tar_txt,
          add_special_tokens=True,
          # Add special tokens, such as [CLS] and [SEP]
          max_length=30,  # Set the maximum length, truncate if exceeded
          pad_to_max_length=True,  # Pad to the maximum length
          return_attention_mask=True,  # Return attention masks
          return_tensors='pt'  # Return tensor type, here is PyTorch
        )

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
            acc += 1
    # print("SNR:",arg.snr," accuracy:",acc/len(src_txts))
    cos_sims = np.array(cos_sims)
    accs = []
    ths = []
    for i in range(60, 90, 2):
        th = i / 100
        ths.append(th)
        acc = cos_sims[cos_sims > th]
        acc = len(acc) / len(cos_sims)
        accs.append(acc)
    print("accs:", accs)

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
    print(training_texts)
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    tr_model = BertModel.from_pretrained(model_name)
    # training SC
    for snr in [0,5,10,15,20,25]:
        arg.snr = snr
        channel_model = channel_net(in_dims=512, snr=arg.snr)
        SC_train(tr_model,channel_model,tokenizer,training_texts, arg)

    # evaluate
    for snr in [0, 5, 10, 15, 20, 25]:
        arg.snr = snr
        channel_model = channel_net(in_dims=512, snr=arg.snr)
        SC_test(tr_model, channel_model, tokenizer, training_texts, arg)

    src_path = r"E:\datasets\VOC2012_img2text"
    res_path = r"E:\datasets\DeepJSCC_res"
    src_txts = [os.path.join(src_path, fname) for fname in os.listdir(src_path) if fname.endswith(".json")]
    res_txts = [os.path.join(res_path, fname) for fname in os.listdir(res_path) if fname.endswith(".json")]
    src_contents = []
    for src_txt in src_txts:
        with open(src_txt, "r", encoding="utf-8") as f:
            src_content = json.load(f)[0]
        src_contents.append(src_content)
    print(src_content)
    res_contents = []
    for res_txt in res_txts:
        with open(res_txt, "r", encoding="utf-8") as f:
            res_content = json.load(f)[0]
        res_contents.append(src_content)
    evaluate(src_contents, res_contents, arg)


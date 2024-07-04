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
from channel_nets import channel_net
import time
import numpy as np
import torchvision
import random

torch.cuda.set_device(0)


class params():
    # Parameters for model configuration and training
    checkpoint_path = "checkpoints"  # Path to save model checkpoints
    device = "cuda"  # Device to use for computation (GPU)
    dataset = r"E:\pengyubo\datasets\UCF100"  # Path to the dataset
    log_path = "logs"  # Path to save logs
    epoch = 5  # Number of training epochs
    lr = 1e-3  # Learning rate
    batchsize = 8  # Batch size for training
    snr = 25  # Signal-to-Noise Ratio
    weight_delay = 1e-5  # Weight decay for regularization
    sim_th = 0.6  # Similarity threshold for evaluation


def same_seeds(seed):
    # Function to set the same random seed for reproducibility
    random.seed(seed)  # Python built-in random module
    np.random.seed(seed)  # Numpy
    torch.manual_seed(seed)  # Torch
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # Torch for CUDA
        torch.cuda.manual_seed_all(seed)  # Torch for all CUDA devices
    torch.backends.cudnn.benchmark = False  # Disable cuDNN benchmark
    torch.backends.cudnn.deterministic = True  # Ensure deterministic results


@torch.no_grad()
def model_test(tr_model, channel_net, tokenizer, training_texts, arg):
    # Function to test the model with a given test set
    tr_model = tr_model.to(arg.device)  # Move the model to the specified device
    tr_model.eval()  # Set the model to evaluation mode
    channel_model = channel_net.to(arg.device)  # Move the channel model to the specified device
    weights_path = f"{arg.checkpoint_path}/t2t_ch_snr{arg.snr}.pth"  # Path to the saved model weights
    weight = torch.load(weights_path, map_location="cpu")  # Load the model weights
    channel_model.load_state_dict(weight)  # Load the weights into the channel model
    channel_model.eval()  # Set the channel model to evaluation mode

    raw_text = []  # List to store raw text
    rec_text = []  # List to store reconstructed text
    random.shuffle(training_texts)  # Shuffle the training texts

    for i in range(0, len(training_texts), arg.batchsize):
        # Process the texts in batches
        if i + arg.batchsize < len(training_texts):
            b_text = training_texts[i:i + arg.batchsize]
        else:
            b_text = training_texts[i:]

        # Tokenize the input text
        input_ids = tokenizer.batch_encode_plus(b_text, return_tensors="pt", padding=True, max_length=512)[
            "input_ids"].to(arg.device)

        # Encode the input text
        encoder_outputs = tr_model.get_encoder()(input_ids)
        model_inputs = {
            "input_ids": None,
            "encoder_outputs": encoder_outputs,
            "past_key_values": None,
        }

        # Generate decoded ids from the model
        decoded_ids = tr_model.generate(**model_inputs)

        # Decode the generated ids to text
        translated_texts = [tokenizer.decode(t, skip_special_tokens=True) for t in decoded_ids]
        raw_text += translated_texts

        shape = encoder_outputs[0].shape
        encoder_outputs_temp = encoder_outputs[0].view(-1, 512)

        # Pass the encoder outputs through the channel model
        encoder_outputs_with_noise = channel_model(encoder_outputs_temp)
        encoder_outputs[0].data = encoder_outputs_with_noise.view(shape).data

        # Generate decoded ids from the noisy encoder outputs
        model_inputs = {
            "input_ids": None,
            "encoder_outputs": encoder_outputs,
            "past_key_values": None,
        }
        decoded_ids = tr_model.generate(**model_inputs)

        # Decode the generated ids to text
        translated_texts = [tokenizer.decode(t, skip_special_tokens=True) for t in decoded_ids]
        rec_text += translated_texts

    # Save the results to a JSON file
    with open(os.path.join(arg.log_path, f"t2t_snr{arg.snr}_eval_res.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps({"raw_text": raw_text, "rec_text": rec_text}, indent=4, ensure_ascii=False))


@torch.no_grad()
def evaluate(src_txts, tar_txts, arg):
    # Function to evaluate the similarity between source and target texts
    from transformers import BertTokenizer, BertModel
    from sklearn.metrics.pairwise import cosine_similarity

    # Load the BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    acc = 0
    cos_sims = []

    for src_txt, tar_txt in zip(src_txts, tar_txts):
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
        cos_sims.append(similarity)

        if similarity > arg.sim_th:
            acc += 1

    cos_sims = np.array(cos_sims)
    accs = []
    ths = []

    for i in range(60, 90, 2):
        th = i / 100
        ths.append(th)
        acc = cos_sims[cos_sims > th]
        acc = len(acc) / len(cos_sims)
        accs.append(acc)

    print("ths:", ths)
    print("accs:", accs)


if __name__ == '__main__':
    same_seeds(1024)
    arg = params()
    with open(os.path.join(arg.log_path, f"t2t_snr{arg.snr}_eval_res.json"), "r", encoding="utf-8") as f:
        content = json.load(f)
    raw_text = content["raw_text"]
    rec_text = content["rec_text"]
    evaluate(raw_text,rec_text)


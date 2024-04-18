import numpy as np


class Config():
    # FL_model_name = "CNN" # mlp,cnn,lstm
    dataset = "mnist" # cifar10、mnist
    num_class = 10
    n_clients = 10
    distribution_alpha = 0.5
    batch_size_for_clients = 128
    test_batch_size = 128
    communication_rounds = 40
    epochs_for_clients = 1
    learning_rate = 1e-3
    weight_delay = 1e-6
    device = "cuda"
    use_ulex = True
    checkpoints_dir = "checkpoints"
    logs_dir = "logs"
    label_data_ratio = 0.1 # max_size = 10000
    # system param
    Interence = 170
    B_u = 1
    N_0 = -174
    P_k = 0.01
    H_k = -50
    delta = 0.023
    error_ratio = 1 - np.exp(-((delta*(Interence+B_u*N_0))/(P_k*H_k)))

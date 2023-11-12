import torch
import os
import numpy as np
import argparse
import pandas as pd
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

from torch.utils.data import DataLoader
from utils.datasetLoader import LeggedMotion_dataset, LeggedMotion_seq_dataset
from utils.trainer import Trainer
from utils.model import FC_model, LSTM, CNN_time, RNNModel
from utils.helpers import split_sequences


def train(args):
    # hyperparams
    lr = 0.0001  
    batch_size = 50
    epochStart = 0
    epochEnd = 2000
    name_variant = 'lstm'

    # load data
    mat = scipy.io.loadmat('Data_Ankle.mat')
    data_np = mat['ankle_data']
    print(f"Data shape: {data_np.shape}, Epochs from {epochStart} to {epochEnd}")
    
    
    ### divide data
    n = len(data_np)
    train_head = int(0.8*n)  # 8:1:1
    val_head = int(0.95*n)
    train_data = data_np[:train_head, :]
    val_data = data_np[train_head:val_head, :]
    test_data = data_np[val_head:, :]
    print(f'Length of data in Train:{len(train_data)}, Val:{len(val_data)}, Test:{len(test_data)}')
    
    ### mini batches for FC model
    # train_batches = DataLoader(LeggedMotion_dataset(data=train_data), batch_size=batch_size,  num_workers=4)
    # val_batches = DataLoader(LeggedMotion_dataset(data=val_data), batch_size=batch_size,  num_workers=4)
    # test_batches = DataLoader(LeggedMotion_dataset(data=test_data), batch_size=batch_size,  num_workers=4)
    # print(f"batches: train-{len(train_batches)}, val-{len(val_batches)}, test-{len(test_batches)}")

    ### sequence data and mini batches for CNN and LSTM model
    steps_in = 9    # 90 ms
    steps_out = 1
    train_d, train_l = split_sequences(train_data, steps_in, steps_out)
    test_d, test_l = split_sequences(test_data, steps_in, steps_out)
    val_d, val_l = split_sequences(val_data, steps_in, steps_out)
    train_batches = DataLoader(LeggedMotion_seq_dataset(data=train_d, labels=train_l), batch_size=batch_size)
    test_batches = DataLoader(LeggedMotion_seq_dataset(data=test_d, labels=test_l), batch_size=batch_size)
    val_batches = DataLoader(LeggedMotion_seq_dataset(data=val_d, labels=val_l), batch_size=batch_size)
    print(f"batches: train-{len(train_batches)}, val-{len(val_batches)}, test-{len(test_batches)}")

    ## different models
    #model = FC_model(input_size=4)
    model = LSTM(num_classes=1, input_size=3, hidden_size=64, num_layers=1, dropout=0)
    #model = RNNModel(num_classes=1, input_size=5, hidden_size=64, num_layers=1, dropout=0)
    # model = CNN_time(input_size=4)

    ## optim
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    ## loss functions
    #loss = torch.nn.MSELoss(reduction='none')
    #loss = torch.nn.L1Loss(reduction="none")
    loss = torch.nn.HuberLoss(reduction="none", delta=1)
    
    # load trainer script and train using fit method
    trainer = Trainer(model, lr, optim, loss, train_batches, val_batches, test_batches, name_variant=name_variant)
    # trainer.restore_checkpoint(986)
    loss = trainer.fit(epochs_start=epochStart, epochs_end=epochEnd)

    # store hyperparameters for reference
    loss.update(
        {
            "params" : "lstm model, 3i,Adam, lr=0.0001, b=50, hu loss, 1k epochs"
        }
    )

    # save different loss as numpy dict
    root_path = os.path.join(os.path.abspath(os.getcwd()), 'results_time')
    os.makedirs(root_path, exist_ok=True )
    file_name = name_variant+'_'+str(epochStart)+'_'+str(epochEnd)+'_data.npy'
    np.save(os.path.join(root_path, file_name), loss)

    # for testing on trained model checkpoint
    # retrain_model_epoch = 200
    # t_loss = trainer.test_model(epoch_num = retrain_model_epoch)
    # t_file_name = name_variant+'_'+str(retrain_model_epoch)+'_testData.npy'
    # np.save(os.path.join(root_path, t_file_name), t_loss)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training of legged Locomotion')
    parser.add_argument('--fp', '-filepath', type=str, default="Data_Ankle.mat", help='file path of data')
    args = parser.parse_args()

    train(args)

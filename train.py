import torch
import os
import numpy as np
import argparse
import pandas as pd
import scipy.io
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from utils.datasetLoader import LeggedMotion_dataset, LeggedMotion_seq_dataset
from utils.trainer import Trainer
from utils.model import FC_model, LSTM, CNN_time
from utils.helpers import split_sequences


def train(args):
    # hyperparams
    lr = 0.01   # 0.001
    batch_size = 64
    epochStart = 0
    epochEnd = 12
    retrain_model_epoch = 100
    name_variant = 'fc'

    # load data
    mat = scipy.io.loadmat('Data_Ankle.mat')
    data_np = mat['ankle_data']
    print(f"Data shape: {data_np.shape}, Epochs from {epochStart} to {epochEnd}")
    
    # divide data
    n = len(data_np)
    train_head = int(0.8*n)  # 7:2:1
    val_head = int(0.95*n)
    train_data = data_np[:train_head, :]
    val_data = data_np[train_head:val_head, :]
    test_data = data_np[val_head:, :]
    print(f'Length of data in Train:{len(train_data)}, Val:{len(val_data)}, Test:{len(test_data)}')
    
    # Mini batches
    train_batches = DataLoader(LeggedMotion_dataset(data=train_data), batch_size=batch_size,  num_workers=4)
    val_batches = DataLoader(LeggedMotion_dataset(data=val_data), batch_size=batch_size,  num_workers=4)
    test_batches = DataLoader(LeggedMotion_dataset(data=test_data), batch_size=batch_size,  num_workers=4)
    print(f"batches: train-{len(train_batches)}, val-{len(val_batches)}, test-{len(test_batches)}")

    # sequence data
    # steps_in = 10   # 100 ms
    # steps_out = 1
    # train_d, train_l = split_sequences(train_data, steps_in, steps_out)
    # test_d, test_l = split_sequences(test_data, steps_in, steps_out)
    # val_d, val_l = split_sequences(val_data, steps_in, steps_out)
    # train_batches = DataLoader(LeggedMotion_seq_dataset(data=train_d, labels=train_l))
    # test_batches = DataLoader(LeggedMotion_seq_dataset(data=test_d, labels=test_l))
    # val_batches = DataLoader(LeggedMotion_seq_dataset(data=val_d, labels=val_l))


    # model , optim, loss function
    model = FC_model()
    #model = LSTM(num_classes=1, input_size=4, hidden_size=512, num_layers=1, dropout=0)
    #model = CNN_time(input_size=4)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss = torch.nn.MSELoss(reduction='none')
    # load trainer 
    trainer = Trainer(model, lr, optim, loss, train_batches, val_batches, test_batches, name_variant=name_variant)
    loss = trainer.fit(epochs_start=epochStart, epochs_end=epochEnd)
    loss.update(
        {
            "params" : "FC model, 5-8-16, Adam"
        }
    )
    # save loss
    np.save(name_variant+'_'+str(epochStart)+'_'+str(epochEnd)+'_data.npy', loss)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training of Room geometry estimator')
    parser.add_argument('--fp', '-filepath', type=str, default="./rirData/ism_100.npy", help='file path of data')
    args = parser.parse_args()

    train(args)

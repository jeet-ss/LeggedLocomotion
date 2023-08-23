import torch
import os
import numpy as np
import argparse
import pandas as pd
import scipy.io
from sklearn.model_selection import train_test_split

from utils.datasetLoader import LeggedMotion_dataset
from utils.trainer import Trainer
from utils.model import FC_model

def train(args):
    # hyperparams
    lr = 0.01
    batch_size = 64
    epochStart = 0
    epochEnd = 1000
    retrain_model_epoch = 100

    # load data
    mat = scipy.io.loadmat('Data_Ankle.mat')
    data = pd.DataFrame(mat['ankle_data'])
    print(f"Data shape: {data.shape}, Epochs from {epochStart} to {epochEnd}")
    
    # divide data
    train_data, rest_data = train_test_split(data, train_size=0.7, random_state=1)
    val_data, test_data = train_test_split(rest_data, test_size=0.333333, random_state=1)

    train_batches = torch.utils.data.DataLoader(LeggedMotion_dataset(data=train_data), batch_size=batch_size,  num_workers=4)
    val_batches = torch.utils.data.DataLoader(LeggedMotion_dataset(data=val_data), batch_size=batch_size,  num_workers=4)
    test_batches = torch.utils.data.DataLoader(LeggedMotion_dataset(data=test_data), batch_size=batch_size,  num_workers=4)
    print(f"batches: train-{len(train_batches)}, val-{len(val_batches)}, test-{len(test_batches)}")

    # model , optim, loss function
    model = FC_model()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss = torch.nn.MSELoss()
    # load trainer 
    trainer = Trainer(model, lr, optim, loss, train_batches, val_batches, test_batches)
    loss = trainer.fit(epochs_start=epochStart, epochs_end=epochEnd)

    # save loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training of Room geometry estimator')
    parser.add_argument('--fp', '-filepath', type=str, default="./rirData/ism_100.npy", help='file path of data')
    args = parser.parse_args()

    train(args)

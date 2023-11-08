import torch
import os
import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# Global variables
LOG_EVERY_N_STEPS = 100

# CUDA variables
USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
device = 'cuda' if USE_CUDA else 'cpu'

class Trainer:
    def __init__(self, 
                #model,      # model to train
                model_no,    # model number for selection
                lr,         # learning rate
                optimizer,  # optimizer to use
                criterion,  # loss function to use
                train_loader=None,  # data loader for training
                val_loader=None,   # data loader for validation
                test_loader=None,    # test data
                early_stopping_patience=100, # early stopping
                name_variant = '',          # string for differenctiating experiments
                ):
        #
        log_dir = './l_'+name_variant
        self.writer = SummaryWriter(log_dir=log_dir)
        #
        self._model = model_no.type(dtype)
        self._optim = optimizer
        self._crit = criterion.type(dtype)
        self.train_batches = train_loader
        self.val_batches = val_loader
        self.test_batches = test_loader
        self._early_stopping_patience = early_stopping_patience
        self.model_path = name_variant+'_checkpoints'

    def save_checkpoint(self, epoch):
        # create path
        
        if os.path.isdir(self.model_path) != True:
            os.makedirs(self.model_path)
        torch.save({'state_dict': self._model.state_dict()}, os.path.join(self.model_path, 'checkpoint_{:03d}.ckp'.format(epoch)))

    def restore_checkpoint(self, epoch):
        checkpoint = torch.load(os.path.join(self.model_path, 'checkpoint_{:03d}.ckp'.format(epoch)), 'cuda' if USE_CUDA else None)
        self._model.load_state_dict(checkpoint['state_dict'])

    def train_step(self, feats, labels, batch_counter,):
        """
            x_batch :
        """
        # reset grad to zero
        if USE_CUDA:
            torch.cuda.empty_cache()
        self._optim.zero_grad()

        
        # foward pass
        pred = self._model(feats)
        loss = self._crit(pred, labels)
        loss = torch.mean(loss)
        # backward pass
        loss.backward()
        self._optim.step()

        return loss.detach().item()

    def train_epoch(self):
        # set training mode
        self._model.train()
        batch_loss = 0
        for idx, b in enumerate(self.train_batches):
            #
            time, ground_reac, trunk_j, hip_j, knee_j, ankle_j, joint_moment = b
            ground_reac = ground_reac.type(dtype).unsqueeze(1)
            trunk_j = trunk_j.type(dtype).unsqueeze(1)
            hip_j = hip_j.type(dtype).unsqueeze(1)
            knee_j = knee_j.type(dtype).unsqueeze(1)
            ankle_j = ankle_j.type(dtype).unsqueeze(1)
            #
            # inp and label for singe time data
            # inp = torch.hstack((ground_reac, trunk_j, knee_j )).unsqueeze(2)
            # joint_moment = joint_moment.type(dtype).unsqueeze(1)
            # inp for time sequence data
            inp = torch.hstack((ground_reac, trunk_j,hip_j, knee_j, ankle_j))#.unsqueeze(2)
            joint_moment = joint_moment.type(dtype)#.unsqueeze(1)
            # for lstms extra dimension change
            inp = torch.swapaxes(inp, 1, 2)

            loss = self.train_step(inp, joint_moment, idx)
            batch_loss += loss
        avg_loss = batch_loss/self.train_batches.__len__()
        return avg_loss

    def val_step(self, feats, labels):
        #
        pred = self._model(feats)
        #
        loss = self._crit(pred, labels)
        loss_mean = torch.mean(loss)
        return loss_mean.detach().item(), pred.detach(), loss.detach()

    def val_epoch(self, batches, mode='val'):
        # eval mode
        self._model.eval()
        # 
        loss_batch = 0
        batch_pred = torch.empty(0)
        batch_labels = torch.empty(0)
        batch_loss = torch.empty(0)
        counter = 0
        with torch.no_grad():
            for idx, batches in enumerate(batches):
                #
                time, ground_reac, trunk_j, hip_j, knee_j, ankle_j, joint_moment = batches
                ground_reac = ground_reac.type(dtype).unsqueeze(1)
                trunk_j = trunk_j.type(dtype).unsqueeze(1)
                hip_j = hip_j.type(dtype).unsqueeze(1)
                knee_j = knee_j.type(dtype).unsqueeze(1)
                ankle_j = ankle_j.type(dtype).unsqueeze(1)

                ### inp and label for singe time data
                # inp = torch.hstack((ground_reac, trunk_j, knee_j )).unsqueeze(2)
                # joint_moment = joint_moment.type(dtype).unsqueeze(1)

                ### inp and label for time sequence data
                inp = torch.hstack((ground_reac, trunk_j,hip_j, knee_j, ankle_j))#.unsqueeze(2)
                joint_moment = joint_moment.type(dtype)#.unsqueeze(1)
                # for lstms extra dimension change
                inp = torch.swapaxes(inp, 1, 2)
                
                loss, pred, loss_t = self.val_step(inp, joint_moment)
                loss_batch += loss
                #print("in val", inp.size(), ground_reac.size(), loss_t.size() )
                #
                batch_pred = torch.cat((batch_pred, pred.cpu()))
                batch_labels = torch.cat((batch_labels, joint_moment.squeeze().cpu()))
                batch_loss = torch.cat((batch_loss, loss_t.cpu()))
                #print("in val2: ", batch_pred.size(), batch_loss.size(), counter)
        avg_loss = loss_batch/self.val_batches.__len__()
        return avg_loss, batch_pred, batch_labels, batch_loss

    def test_model(self, epoch_num):
        '''
        epoch number is checkpoint number to load mmodel
        '''
        # restore checkpoint
        self.restore_checkpoint(epoch_num)
        # test 
        test_loss, test_pred, test_label, test_errors_B = self.val_epoch(self.test_batches, mode='test')

        return  {
            "test_loss" : test_loss,
            "test_pred" : test_pred.squeeze().numpy(),
            "test_label" : test_label.numpy(),
            "test_errors" : test_errors_B.squeeze().numpy(),
        }

    def fit(self, epochs_start=1, epochs_end=0):
        epochs = epochs_end - epochs_start
        assert epochs > 0, 'Epochs > 0'
        #
        loss_train = np.array([])
        loss_val = np.array([])
        test_loss = 0
        min_loss = np.Inf
        criteria_counter = 0
        preds = np.array([])
        labels = np.array([])
        errors = np.array([])
        #
        #while (True):
        for i in range(epochs_start, epochs_end):

            train_loss = self.train_epoch()
            val_loss, pred, label, errors_B  = self.val_epoch(self.val_batches)

            #
            loss_train = np.append(loss_train, train_loss)
            loss_val = np.append(loss_val, val_loss)
            preds = np.append(preds, pred)
            labels = np.append(labels, label)
            errors = np.append(errors, errors_B)
            #logging
            if i%2 == 0:
                print(f'at epoch {i}, trainLoss:{train_loss:.2f}, valLoss:{val_loss:.2f}')
            self.writer.add_scalar(tag="train_loss", scalar_value=train_loss, global_step=i)
            self.writer.add_scalar("val_loss", val_loss, i)
            # save model if better
            if train_loss < min_loss:
                criteria_counter = 0
                min_loss = train_loss
                # save checkpoint
                self.save_checkpoint(i)
            
            else:
                criteria_counter += 1            
            # early stopping
            if criteria_counter > self._early_stopping_patience:
                print(f"Early Stopping Criteria activated at epoch {i} with criterion counter at {criteria_counter}")
                break
        
        # run test batch
        test_loss, test_pred, test_label, test_errors_B = self.val_epoch(self.test_batches, mode='test')

            
        return  {
            "train_loss" : loss_train,
            "test_loss" : test_loss,
            "test_pred" : test_pred.squeeze().numpy(),
            "test_label" : test_label.numpy(),
            "test_errors" : test_errors_B.squeeze().numpy(),
            "val_loss" : loss_val,
            "min_loss" : min_loss,
            "val_preds" : preds,
            "val_labels" : labels,
            "val_errors" : errors,
        }
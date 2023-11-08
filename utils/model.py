import torch
import torch.nn as nn


class FC_model(nn.Module):
    """
        This model estimates 1 joint moment
    """
    def __init__(self, input_size=5):
        super().__init__()
        # define layers
        self.fc = nn.Sequential(
                        nn.Linear(input_size, 16), nn.ReLU(),       #6
                        nn.Linear(16, 32), nn.ReLU(),    
                        nn.Linear(32, 16), nn.ReLU(),        #8
                        nn.Linear(16, 1)                     #9
                    )
        
    def forward(self, inp):
        return self.fc(inp.squeeze())

class RNNModel(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, dropout=0.2):
        super().__init__()
        self.num_classes = num_classes # output size
        self.num_layers = num_layers # number of recurrent layers in the lstm
        self.input_size = input_size # input size
        self.hidden_size = hidden_size # neurons in each lstm layer

        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc_1 =  nn.Linear(hidden_size, 32) # fully connected 
        self.fc3 = nn.Linear(32,8)
        self.fc_2 = nn.Linear(8, num_classes) # fully connected last layer
        self.relu = nn.ReLU()

    def forward(self,x):
        # hidden state
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()
        # cell state
        #c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()
        # propagate input through LSTM
        output, hn = self.rnn(x, h_0) # (input, hidden, and internal state)
        #print('before:', hn.size())
        hn = hn.view(-1, self.hidden_size) # reshaping the data for Dense layer next
        #print('after:', hn.size())
        out = self.relu(hn)
        out = self.fc_1(out) # first dense
        out = self.relu(out) # relu
        out = self.fc3(out) # second dense
        out = self.relu(out)
        out = self.fc_2(out) # final output
        return out


class LSTM(nn.Module):
    
    def __init__(self, num_classes, input_size, hidden_size, num_layers, dropout=0.2):
        super().__init__()
        self.num_classes = num_classes # output size
        self.num_layers = num_layers # number of recurrent layers in the lstm
        self.input_size = input_size # input size
        self.hidden_size = hidden_size # neurons in each lstm layer
        # LSTM model
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout) # lstm
        self.fc_1 =  nn.Linear(hidden_size, 32) # fully connected 
        self.fc3 = nn.Linear(32,8)
        self.fc_2 = nn.Linear(8, num_classes) # fully connected last layer
        self.relu = nn.ReLU()
        
    def forward(self,x):
        # hidden state
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()
        # cell state
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()
        # propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) # (input, hidden, and internal state)
        #print('before:', hn.size())
        hn = hn.view(-1, self.hidden_size) # reshaping the data for Dense layer next
        #print('after:', hn.size())
        out = self.relu(hn)
        out = self.fc_1(out) # first dense
        out = self.relu(out) # relu
        out = self.fc3(out) # second dense
        out = self.relu(out)
        out = self.fc_2(out) # final output
        return out
    
class Conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, ):
        super().__init__()

        self.seq = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=False),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU()

                )
        
    def forward(self, input_1D):
        return self.seq(input_1D)
    
class CNN_dense(nn.Module):
    def __init__(self, input_size, channels=[32,64,],  kernel_size=3, dropout=0.2):
        super().__init__()
        """
            Input size = No of channels / no of Input features used
        """
        self.c1 = Conv_block(input_size, channels[0], kernel_size=1)
        self.c2 = Conv_block(channels[0], channels[1], kernel_size=1)
        self.seq = nn.Sequential(
                    nn.Linear(64,32), nn.ReLU(),
                    nn.Linear(32,8), nn.ReLU(),
                    nn.Linear(8,1)

        )
    def forward(self, inp):
        batch_size = inp.size()[0]
        inp = self.c1(inp)
        inp = self.c2(inp)
        inp = inp.reshape(batch_size, -1)
        inp = self.seq(inp)
        
        return inp


class CNN_time(nn.Module):
    def __init__(self, input_size, channels=[32,64],  kernel_size=3, dropout=0.2):
        super().__init__()

        self.c1 = Conv_block(input_size, channels[0], kernel_size=3)
        self.c2 = Conv_block(channels[0], channels[1], kernel_size=3)
        self.mp = nn.MaxPool1d(kernel_size=2)
        self.seq = nn.Sequential(
                    nn.Linear(64,32), nn.ReLU(),
                    nn.Linear(32,8), nn.ReLU(),
                    nn.Linear(8,1)

        )
    def forward(self, inp):
        inp = self.mp(self.c1(inp))
        #
        #print("for1:", inp.size())
        inp = self.mp(self.c2(inp))
        #print("for2:", inp.size())
       
        inp = self.seq(inp.reshape(inp.size()[0], -1))
        #print("for3:" , inp.size())

        return inp

class CNN_time_2(nn.Module):
    '''
        this model has no maxpool
    '''
    def __init__(self, input_size, channels=[16,32,64],  kernel_size=3, dropout=0.2):
        super().__init__()

        self.c1 = Conv_block(input_size, channels[0], kernel_size=5)
        self.c2 = Conv_block(channels[0], channels[1], kernel_size=3)
        self.c3 = Conv_block(channels[1], channels[2], kernel_size=3)
        #self.mp = nn.MaxPool1d(kernel_size=2)
        self.seq = nn.Sequential(
                    nn.Linear(64,32), nn.ReLU(),
                    nn.Linear(32,8), nn.ReLU(),
                    nn.Linear(8,1)

        )
    def forward(self, inp):
        inp = self.c1(inp)
        #
        #print("for1:", inp.size())
        inp = self.c2(inp)
        #print("for2:", inp.size())
        inp = self.c3(inp)
        #print("for3:", inp.size())
        inp = self.seq(inp.reshape(inp.size()[0], -1))
        #print("for4:" , inp.size())

        return inp
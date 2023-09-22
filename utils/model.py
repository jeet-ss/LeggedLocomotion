import torch
import torch.nn as nn


class FC_model(nn.Module):
    """
        This model estimates 1 joint moment
    """
    def __init__(self):
        super().__init__()
        # define layers
        self.fc = nn.Sequential(
                        nn.Linear(5, 16), nn.ReLU(),       #6
                        nn.Linear(16, 32), nn.ReLU(),    
                        nn.Linear(32, 16), nn.ReLU(),        #8
                        nn.Linear(16, 1)                     #9
                    )
        
    def forward(self, inp):
        return self.fc(inp.squeeze())

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
        self.fc_1 =  nn.Linear(hidden_size, 256) # fully connected 
        self.fc3 = nn.Linear(256,32)
        self.fc_2 = nn.Linear(32, num_classes) # fully connected last layer
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
        out = self.fc3(out) # first dense
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

class CNN_time(nn.Module):
    def __init__(self, input_size, channels=[32,64,128,256],  kernel_size=3, dropout=0.2):
        super().__init__()

        self.c1 = Conv_block(input_size, channels[0], kernel_size=1)
        self.c2 = Conv_block(channels[0], channels[1], kernel_size=1)
        self.c3 = Conv_block(channels[1], channels[2], kernel_size=3)
        self.c4 = Conv_block(channels[2], channels[3], kernel_size=2)
        self.mp = nn.MaxPool1d(kernel_size=2)
        self.seq = nn.Sequential(
                    #nn.Linear(256,64), nn.ReLU(),
                    nn.Linear(64,32), nn.ReLU(),
                    nn.Linear(32,8), nn.ReLU(),
                    nn.Linear(8,1)

        )
    def forward(self, inp):
        inp = self.c1(inp)
        #
        #print("for:", inp.size())
        inp = self.c2(inp)
        #print("for:", inp.size())
        #inp = self.c3(inp)

        #inp = self.mp(inp)
        #inp = self.c4(inp)

        #inp = self.mp(inp)
        #print("for:", inp.size())
        inp = self.seq(inp.reshape(inp.size()[0], -1))

        return inp
import torch.nn as nn

class FC_model(nn.Module):
    """
        This model estimates 1 joint moment
    """
    def __init__(self):
        super().__init__()
        # define layers
        self.fc = nn.Sequential(
                        nn.Linear(4, 8), nn.ReLU(),       #6
                        nn.Linear(8, 16), nn.ReLU(),       #7
                        nn.Linear(16, 8), nn.ReLU(),        #8
                        nn.Linear(8, 1)                     #9
                    )
        
    def forward(self, inp):
        return self.fc(inp)
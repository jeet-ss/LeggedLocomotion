import torch

class LeggedMotion_dataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        """
         7 different data, being time, 
         vertical ground reaction force in Newton, 
         trunk joint angle, hip joint angle,
         knee joint angle, ankle joint angle (all in radians),
         and the joint moment in Nm.
        """
        self.transform = transform
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 
        features = torch.tensor(self.data.iloc[idx])
        #print("features:", features.shape)
        # 
        return (features[0], features[1], features[2], features[3], features[4], features[5], features[6])

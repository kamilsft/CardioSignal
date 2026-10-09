import os
import torch
from torch.utils.data import Dataset

class dataset(Dataset):
    def __init__(self, data_folders):
        self.file_names = []

        # Using two data folders to load the data
        for data_folder in data_folders:
            all_files = os.listdir(data_folder)
            for file in all_files:
                if file.endswith('.pt'):
                    file_path = os.path.join(data_folder, file)
                    data = torch.load(file_path)
                    if isinstance(data, list):
                        for segment in data:
                            if isinstance(segment, dict) and 'ecg' in segment and 'ppg' in segment:
                                self.file_names.append(segment)
        
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, index):
        segment = self.file_names[index]
        ecg = torch.tensor(segment['ecg'], dtype=torch.float32)  
        pcg = torch.tensor(segment['ppg'], dtype=torch.float32)

        # ensure the shape is (1, 48000) for both ECG and PCG
        ecg = ecg.view(1, -1)
        pcg = pcg.view(1, -1)
        return ecg, pcg
    
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    pcgecgdataset = dataset('YourPATH')
    # we need to add more path for unhealthy data.
    pcgecgdataset_unhealthy = dataset('YourPATH')
    loader = DataLoader(pcgecgdataset, batch_size=4, shuffle=True)
    loader2 = DataLoader(pcgecgdataset_unhealthy, batch_size=4, shuffle=True)
    for ecg, pcg in loader:
        print("ECG batch shape:", ecg.shape)
        print("PCG batch shape:", pcg.shape)
        break  # just to test the first batch
    for ecg, pcg in loader2:
        print("ECG batch shape:", ecg.shape)
        print("PCG batch shape:", pcg.shape)
        break
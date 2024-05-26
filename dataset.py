# import some packages you need here
import torch
from torch.utils.data import Dataset

class Shakespeare(Dataset):
    """ Shakespeare dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        input_file: txt file

    Note:
        1) Load input file and construct character dictionary {index:character}.
           You need this dictionary to generate characters.
        2) Make list of character indices using the dictionary
        3) Split the data into chunks of sequence length 30. 
           You should create targets appropriately.
    """

    def __init__(self, input_file):
        # Load input file
        with open(input_file, 'r') as f:
            self.data = f.read()

        # Construct character dictionary
        self.chars = sorted(set(self.data))
        self.char2idx = {char: idx for idx, char in enumerate(self.chars)}
        self.idx2char = {idx: char for idx, char in enumerate(self.chars)}

        # Make list 
        self.data_indices = [self.char2idx[char] for char in self.data]
        self.seq_length = 30

        # Prepare input & target sequences
        self.inputs = []
        self.targets = []
        for i in range(0, len(self.data_indices) - self.seq_length):
            self.inputs.append(self.data_indices[i:i + self.seq_length])
            self.targets.append(self.data_indices[i + 1:i + self.seq_length + 1])

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        inputs= torch.tensor(self.inputs[idx], dtype=torch.long)
        targets = torch.tensor(self.targets[idx], dtype=torch.long)
        return inputs, targets

if __name__ == '__main__':
    # Test codes to verify the implementations
    dataset = Shakespeare('./shakespeare_train.txt')
    print("Dataset size:", len(dataset))
    input_sample, target_sample = dataset[0]
    print("Sample data (input):", input_sample)
    print("Sample data (target):", target_sample)

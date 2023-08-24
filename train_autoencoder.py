import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import glob
from typing import Tuple

from model import measure_size, Autoencoder, the_dtype, categories

batch_size = 64
learning_rate = 0.001
num_epochs = 1000

silence = b'\xff'*measure_size

device = torch.device("cpu")
#device = torch.device("mps")

class MusicDataset(Dataset):
    def __init__(self):
        self.measures = []
        self.cache = {}

    def add_file(self, path):
        with open(path, 'rb') as f:
            silences = 0
            while True:
                data = f.read(measure_size)
                if not data:
                    break
                if data == silence:
                    silences += 1
                else:
                    silences = 0

                if silences < 4:
                    self.measures.append(data)

    def __len__(self):
        return len(self.measures)-3

    def __getitem__(self, idx):
        chunk = self.cache.get(idx)
        if chunk is None:
            chunk = torch.tensor(tuple(self.measures[idx] + self.measures[idx+1] + self.measures[idx+2] + self.measures[idx+3]), dtype=torch.uint8, device=device)
            rests = chunk == 255
            chunk = chunk * (~rests)
            chunk = chunk + (rests * 39)
            self.cache[idx] = chunk
        return (chunk, 0)

train_dataset = MusicDataset()
for example in glob.iglob('train_data/*'):
    train_dataset.add_file(example)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

print("data loaded")

# Initialize the model and loss function
autoencoder = Autoencoder().to(device)
#criterion = nn.MSELoss()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)

@torch.jit.script
def prepare_inputs(inputs: torch.Tensor, blank_prob: float, num_categories: int) -> Tuple[torch.Tensor, torch.Tensor]:
    blank_mask = torch.rand_like(inputs.float()) < blank_prob
    non_blank_mask = ~blank_mask

    blanked_inputs = (inputs * non_blank_mask) + (38 * blank_mask)
    blanked_train_inputs = nn.functional.one_hot(blanked_inputs, num_categories).float()
    train_inputs = nn.functional.one_hot(inputs, num_categories).float()

    outputs = autoencoder(blanked_train_inputs).softmax(dim=2)

    return train_inputs, outputs


def train_nn(epoch, blank_prob):
    for batch_i, batch_data in enumerate(train_loader):
        inputs, _ = batch_data
        
        optimizer.zero_grad()

        train_inputs, outputs = prepare_inputs(inputs, blank_prob, categories)

        loss = criterion(outputs, train_inputs)
        loss.backward()
        optimizer.step()

        if batch_i % 400 == 0:
            print("  Batch [" + str(batch_i) + "], Loss: " + str(loss.item()))
    
    return loss

# Training loop
blank_prob = 0.0
for epoch in range(num_epochs):
    loss = train_nn(epoch, 0.2)

    #if blank_prob < 0.25:
    #    blank_prob += 0.01

    if epoch % 5 == 0:
        torch.save({'epoch': epoch, 'model_state_dict': autoencoder.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': loss}, f"model/chkpt_{epoch}")

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")



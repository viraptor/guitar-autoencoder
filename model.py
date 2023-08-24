import torch
import torch.nn as nn

first_mid_size = 128
second_mid_size = 64
third_mid_size = 64

measure_size = 6*16
categories = 40

the_dtype = torch.float32

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(measure_size*4*categories, first_mid_size, dtype=the_dtype),
            nn.ReLU(True),
            nn.Linear(first_mid_size, second_mid_size, dtype=the_dtype),
            #nn.ReLU(True),
            #nn.Linear(second_mid_size, third_mid_size, dtype=the_dtype),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            #nn.Linear(third_mid_size, second_mid_size, dtype=the_dtype),
            #nn.ReLU(True),
            nn.Linear(second_mid_size, first_mid_size, dtype=the_dtype),
            nn.ReLU(True),
            nn.Linear(first_mid_size, measure_size*4*categories, dtype=the_dtype),
            nn.Sigmoid(),
            nn.Unflatten(1, (measure_size*4, categories)),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

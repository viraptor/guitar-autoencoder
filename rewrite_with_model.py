import sys
import torch
from model import measure_size, Autoencoder, categories

autoencoder = Autoencoder()
checkpoint = torch.load(sys.argv[1])
autoencoder.load_state_dict(checkpoint['model_state_dict'])

with open(sys.argv[2], 'rb') as f:
    orig_data = f.read()


#with open(sys.argv[3], 'wb') as f:
for start in range(0, len(orig_data)//measure_size - 3, 4):
    data_start = start * measure_size
    data_end = (start+4) * measure_size
    gap_start = 2 * measure_size
    gap_end = 3 * measure_size

    input_data = torch.tensor(tuple(orig_data[data_start:data_end]), dtype=torch.uint8)
    rests = input_data == 255
    input_data = input_data * (~rests)
    input_data = input_data + (rests * 39)

    input_data[gap_start:gap_end] = 38

    print("Input")
    print(input_data)
    input_data = torch.nn.functional.one_hot(input_data, categories)
    output_data = autoencoder(input_data.float().view(1, measure_size*4, categories))

    print("Output")
    print(torch.argmax(output_data, 2))

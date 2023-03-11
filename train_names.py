from torch.utils.data import DataLoader
from torch.optim import AdamW
from makemore import ModelConfig, create_datasets, Bigram, generate, MLP
from tqdm import tqdm
import numpy as np
import torch

def string_to_input(string, dataset):
    return torch.tensor([[dataset.stoi[char] for char in string]])

def output_to_string(output, dataset):
    strings = []
    for word in output:
        strings.append(''.join([dataset.itos[char.item()] for char in word if char > 0]))
    return strings


# Hyperparameters
learning_rate = 0.0005
weight_decay = 0.01
num_epochs = 10

# load data and make dataloaders
names_file = 'names.txt'
train_set, test_set = create_datasets(names_file, num_words=None)
train_loader = DataLoader(train_set, shuffle=True)
test_loader = DataLoader(test_set)

# make model and optimizer
config = ModelConfig(block_size=1, vocab_size=train_set.get_vocab_size(),
                     n_embd=8, n_embd2=256)
# model = Bigram(config)
model = MLP(config)
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay,
                              betas=(0.9, 0.99), eps=1e-8)

# train
for epoch in range(num_epochs):
    losses = []

    for x, target in tqdm(train_loader):
        model.zero_grad()

        preds, loss = model(x, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    print(f'Loss in epoch {epoch}: {np.mean(losses).item()}')

model.eval()
# maak Beerends naam af
beer = [[train_set.stoi[letter] for letter in 'beer'], [train_set.stoi[letter] for letter in 'timo']]
batch_of_beers = torch.tensor(beer)#.unsqueeze(0).repeat(3, 1)

# print(model.logits.max())

batch_of_beers = torch.zeros((2, 1), dtype=torch.long)
generated_words = generate(model, batch_of_beers, 10, top_k=10, do_sample=True)

print(output_to_string(generated_words, train_set))



import torch
from torch import nn
import numpy as np
import torchsummary
import time
import read_wav

# This file implements Baghel's CNN attempt to classify PCG samples, which we set as benchmark.
# For their paper, see https://www.sciencedirect.com/science/article/abs/pii/S0169260720315832.

def init_weights(module: nn.Module) -> None:
    """initialize weights for nn params."""
    if type(module) == nn.Linear or type(module) == nn.Conv1d:
        nn.init.xavier_uniform_(module.weight)

def accuracy(test_samples: list[torch.Tensor, torch.Tensor], net: nn.modules.container.Sequential) -> float:
    """find model's accuracy on test samples."""
    [test_features, test_labels] = test_samples
    with torch.no_grad():
        accu = (net(test_features.to(torch.float32).reshape(-1, 1, 20000)).argmax(axis=1) == test_labels).sum() / len(test_features)
    return accu

# model initialization.
net = nn.Sequential(
    nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=2),
    nn.ReLU(),
    nn.BatchNorm1d(num_features=64, ),
    nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=2),
    nn.ReLU(),
    nn.BatchNorm1d(num_features=64),
    nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=2),
    nn.ReLU(),
    nn.BatchNorm1d(num_features=64),
    nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=2),
    nn.ReLU(),
    nn.BatchNorm1d(num_features=64),
    nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=2),
    nn.ReLU(),
    nn.BatchNorm1d(num_features=64),
    nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=3),
    nn.ReLU(),
    nn.BatchNorm1d(num_features=64),
    nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=2),
    nn.ReLU(),
    nn.BatchNorm1d(num_features=32),
    nn.MaxPool1d(kernel_size=2),
    nn.Dropout(p=0.15),
    nn.Flatten(start_dim=1),
    nn.Linear(in_features=51*32, out_features=128),
    nn.Dropout(p=0.30),
    nn.Linear(in_features=128, out_features=5),
    nn.Softmax()
) # model structure.

lr = 0.02
batch_size = 40
num_epoches = 100


net.apply(init_weights) # generate uniformly distributed initial weight for params.
optimizer = torch.optim.SGD(net.parameters(), lr=lr) # apply SGD as learning algorithm.
loss = nn.CrossEntropyLoss() # apply cross entropy loss as loss function.

# load sample datas, transform them into tensor type, and divide the training samples into batches.
[train_features, train_labels, test_features, test_labels] = [torch.tensor(np.array(list_)) for list_ in read_wav.load_data()]
train_batches = [[train_features[idx:idx+batch_size].reshape(batch_size, 1, 20000), train_labels[idx:idx+batch_size]] for idx in range(0, len(train_features), batch_size)]

torchsummary.summary(net, (1, 20000))

total_train_time = 0

for epoch in range(num_epoches):
    for batch_idx, (features, labels) in enumerate(train_batches):
        train_start_time = time.time()
        optimizer.zero_grad() # reset param gradients to zero.
        result = net(features.to(torch.float32))
        l = loss(net(features.to(torch.float32)), labels).sum() # obtain loss for this batch.
        l.backward() # obtain gradiant through backward propagation.
        optimizer.step() # update params.
        with torch.no_grad():
            batch_train_time = time.time() - train_start_time
            total_train_time += batch_train_time
            test_start_time = time.time()
            accu = accuracy([test_features, test_labels], net) # calculate classification accuracy on test samples.
            print(f"epoch {epoch} step {batch_idx}:\naccuracy: {accu * 100: .2f}%\nloss: {l: .2f}\ntotal train time: {total_train_time: .2f}s\ntest time for 200 samples: {time.time() - test_start_time: .2f}s\n\n")



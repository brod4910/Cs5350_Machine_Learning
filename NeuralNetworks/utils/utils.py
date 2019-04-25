import numpy as np
import torch
import torch.nn.functional as F

def prepare_continous_data(f_name):
    examples = []
    labels = []
    with open(f_name) as f:
        for line in f:
            s = []
            sample = line.strip().split(',')
            for i, item in enumerate(sample[:-1]):
                s.append(float(item))
            s.append(1)
            examples.append(s)
            labels.append([float(sample[-1])])
    return np.array(examples), np.array(labels)

def shuffle_data(X,Y):
    sample = [i for i in range(len(Y))]
    np.random.shuffle(sample)
    
    x_s = np.array([X[s] for s in sample])
    y_s = np.array([Y[s] for s in sample])

    return x_s, y_s

def train(model, optimizer, criterion, data_loader, epoch):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    for batch_idx, data in enumerate(data_loader):
        out = model(data['input'])

        loss = criterion(out, data['label'])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / (batch_idx+1)


def test(model, data_loader, epoch):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            out = model(data['input'])

            total_loss += F.cross_entropy(out, data['label'])

    
    return total_loss/ (batch_idx + 1)    
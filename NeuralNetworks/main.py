# local imports
from pytorch_nn import NeuralNetwork as torch_nn
from BankNoteDataset import BankNoteDataset
from neural_network import NeuralNetwork
from utils.utils import prepare_continous_data, shuffle_data, train, test
# library imports
import torch
import torch.nn.functional as F
import numpy as np

def pytorch_model():
    depth = [3, 5, 9]
    hidden_nodes = [5, 10, 25, 50, 100]
    init_params = [['tanh', 'xavier'], ['relu', 'he']]
    epoch = 10

    train_dataset = BankNoteDataset('./bank-note/train.csv', 4)
    test_dataset = BankNoteDataset('./bank-note/test.csv', 4)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size= 12,
        shuffle= True,
        num_workers= 6,
        pin_memory= True
        )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size= 12,
        shuffle= True,
        num_workers= 6,
        pin_memory= True
        )
    
    for (activation, initialization) in init_params:
        for d in depth:
            for nodes in hidden_nodes:
                print('Depth:', d, ', Nodes:', nodes, ', Activation:', activation, ', Initialization:', initialization)
                model = torch_nn(4, 2,  nodes, d, activation, initialization)
                optimizer = torch.optim.Adam(model.parameters())
                criterion = torch.nn.CrossEntropyLoss()
                train_loss = 0
                test_loss = 0
                for epoch in range(1, epoch + 1):
                    train_loss += train(model, optimizer, criterion, train_loader, epoch)
                    test_loss += test(model, test_loader, epoch)

                print('Train Loss: {:.6f}'.format(train_loss))
                print('Test Loss: {:.6f}'.format(test_loss))


def python_model():
    train_ex, train_labels = prepare_continous_data('./bank-note/train.csv')
    test_ex, test_labels = prepare_continous_data('./bank-note/test.csv')
    train_labels = np.array([[-1] if l == 0 else [l] for l in train_labels])
    test_labels = np.array([[-1] if l == 0 else [l] for l in test_labels])
    
    epochs = 25
    lr = .001
    dp = [100/873, 500/873, 700/873]
    nodes = [5, 10, 25, 50, 100]
    num_features = 5

    for hidden_nodes in nodes:
        for d in dp:
            print('Number of hidden nodes: {}\t Parameter d: {}'.format(hidden_nodes, d))
            network = NeuralNetwork(num_features, hidden_nodes, lr, d)
            train_loss = 0
            for epoch in range(1, epochs + 1):
                X, Y = shuffle_data(train_ex, train_labels)

                train_loss += network.train_dataset(X, Y, epoch)
                # print('Epoch: {}\n\t Train Loss: {:.6f}'.format(epoch, loss[0,0]))
            train_err = network.test_dataset(train_ex, train_labels)
            test_err = network.test_dataset(test_ex, test_labels)

            print('Train Loss: {:.6f}'.format(train_err))
            print('Test Error: {:.6f}'.format(test_err))

if __name__ == '__main__':
    pytorch_model()
    # python_model()
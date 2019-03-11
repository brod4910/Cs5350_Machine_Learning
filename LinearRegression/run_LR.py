import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from utils.prepare_data import prepare_data, prepare_continous_data
from LinearRegressor import LinearRegressor
import numpy as np
import matplotlib.pyplot as plt

def main():
    # attr_names = ['Cement', 'Slag', 'Fly ash', 'Water', 'SP', 'Coarse Aggr', 'Fine Aggr']
    train_examples, train_labels = prepare_continous_data('./concrete/train.csv')
    test_examples, test_labels = prepare_continous_data('./concrete/test.csv')

    weights = np.zeros((train_examples.shape[1], 1))
    lr = .001
    epochs = 100


    print('Training step\t|\tTrain Cost function \t|\tConvergence\t|\tTest Cost Function\t|')
    train_error = []
    test_error = []

    regressor = LinearRegressor(lr, weights)
    for epoch in range(1, epochs+1):
        lms_train, convergence = regressor.train(train_examples, train_labels)

        train_error.append(lms_train)
        preds = regressor.test_batch(test_examples, test_labels)
        lms_test = regressor._calc_error(preds, test_labels)

        test_error.append(lms_test)
        print('{}\t\t|\t{:.6f}\t\t|\t{:.6f}\t|\t{:.6f}\t|'.format(epoch, lms_train, np.sum(convergence), lms_test))
    print('Final Weight Vector:\n', regressor.weights)
    print('Learning rate: ', lr)


    sample = [i for i in range(len(train_examples))]
    np.random.shuffle(sample)

    print("\nRUNNING SGD")
    print('Training step\t|\tExample\t\t\t|\tConvergence\t|')
    train_error = []
    weights = np.zeros((train_examples.shape[1], 1))

    regressor = LinearRegressor(lr, weights)
    for epoch in range(1, epochs+1):
        for i, s in enumerate(sample):
            lms_train, convergence = regressor.train(train_examples[s].reshape((1, train_examples.shape[1])), train_labels[s])

            train_error.append(lms_train)

            print('{}\t\t|\t{} ({})\t|\t{:.6f}\t|'.format(epoch, i, s, lms_train))


        preds = regressor.test_batch(test_examples, test_labels)
        lms_test = regressor._calc_error(preds, test_labels)

    print('Final Weight Vector:\n', regressor.weights)
    print('Learning rate: ', lr)
    print('Test error: ', lms_test)


if __name__ == '__main__':
    main()
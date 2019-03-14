import numpy as np
def prepare_continous_data(f_name):
    examples = []
    labels = []
    with open(f_name) as f:
        for line in f:
            s = []
            sample = line.strip().split(',')
            for i, item in enumerate(sample[:-1]):
                s.append(float(item))
            examples.append(s)
            labels.append([float(sample[-1])])
    return np.array(examples), np.array(labels)

def shuffle_data(X,Y):
    sample = [i for i in range(len(Y))]
    np.random.shuffle(sample)
    
    x_s = np.array([X[s] for s in sample])
    y_s = np.array([Y[s] for s in sample])

    return x_s, y_s
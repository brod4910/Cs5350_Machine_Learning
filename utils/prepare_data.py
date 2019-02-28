import numpy as np

def prepare_data(f_name, attr_names, attr_numeric= None, attr_unknown= None):
    examples = []
    labels = []
    attributes = {}
    with open(f_name) as f:
        for line in f:
            s = {}
            sample = line.strip().split(',')
            for i, item in enumerate(sample[:-1]):
                s[attr_names[i]] = item
            examples.append(s)
            labels.append(sample[-1])

    if attr_numeric:
        medians = [[] for __ in range(len(attr_numeric))]

        for s in examples:
            for i, attr in enumerate(attr_numeric):
                num = float(s[attr])
                medians[i].append(num)

        med = [np.median(median) for median in medians]

        for (attr, median) in zip(attr_numeric, med):
            for s in examples:
                # print(attr)
                s[attr] = 'bigger' if float(s[attr]) >= float(median) else 'less'

    if attr_unknown:
        unknowns = [[] for __ in range(len(attr_unknown))]

        for s in train_examples:
            for i, unknown in enumerate(attr_unknown):
                unknowns[i].append(s[unknown])
    
        unknowns = [Counter(unknown).most_common(1)[0][0] for unknown in unknowns]

        for (attr, unknown) in zip(attr_unknown, unknowns):
            for s in examples:
                s[attr] = unknown

    for ex in examples:
        for j, item in enumerate(ex):
            attrs = ex[item]
            if item not in attributes:
                attributes[item] = []
            if attrs not in attributes[item]:
                attributes[item].append(attrs)

    return examples, labels, attributes

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


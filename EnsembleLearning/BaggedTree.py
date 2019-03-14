from copy import deepcopy
from BaseDecisionTree import BaseDecisionTree, Node
from utils.prepare_data import _shuffle_with_replacement

import numpy as np

class BaggedTree(BaseDecisionTree):
    def __init__(self, error_function, num_trees, depth):
        super().__init__(error_function, depth)
        self.num_trees = num_trees
        self.hypotheses = []
        self.hypotheses_votes = []

    '''
        Trains the decision tree trained on the given data
        params:
            examples: examples to base the tree from,
                options: (list of maps, where maps contain keys with each attribute)
            attributes: list of attributes of the training data,
                options: (map: keys = attribute, val = value an attribute can take)
            labels: labels that correspond to the examples in S, 
                options: (list)
    '''
    def train_dataset(self, examples, attributes, labels):
        for n in range(self.num_trees):
            n_examples, n_labels = _shuffle_with_replacement(examples, labels)
            root = self._build_tree(n_examples, attributes, n_labels)

            vote = self._cast_vote(root, n_examples, n_labels)

            self.hypotheses.append(deepcopy(root))
            self.hypotheses_votes.append(deepcopy(vote))

    '''
        Tests the tree with the given examples and labels. Returns the predictions and error
        params:
            examples: data to test the tree,
                options: (list of maps, where maps contain keys with each attribute)
            labels: labels that correspond to the examples in S, 
                options: (list)
        return:
            returns predictions and error
    '''
    def test_dataset(self, examples, labels):
        final_hypoth = np.zeros(len(labels))
        for h, vote in zip(self.hypotheses, self.hypotheses_votes):
            final_hypoth += vote * self._test(h, examples)

        final_hypoth = np.sign(final_hypoth)
        error = self._test_error(final_hypoth, labels)

        return final_hypoth, error

    def _build_tree(self, S, attributes, labels):
        root = self._ID3(S, attributes, labels, self.depth)
        return root

    def _test(self, root, S):
        preds = np.zeros(len(S))
        for i, s in enumerate(S):
            preds[i] = self._prediction(root, s)
        return preds

    def _cast_vote(self, root, examples, labels):
        predicted = self._test(root, examples)

        error = self._test_error(predicted, labels)
        return np.log2((1-error)/(2*error))

    def _prediction(self, root, example):
        while root.children:
            attribute = example[root.attribute]
            if attribute in root.children:
                root = root.children[attribute]

        return root.label



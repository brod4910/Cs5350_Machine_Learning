from copy import deepcopy
import random
from BaseDecisionTree import BaseDecisionTree, Node
from utils.prepare_data import _shuffle_with_replacement

import numpy as np

class RandomForest(BaseDecisionTree):
    '''
        Decision Tree created using ID3 algorithm
        params:
            error_function: The function which to choose the 
                splits of attributes in the data, options: (function)
            depth: depth in which the tree should take, options: (int)

    '''
    def __init__(self, error_function, feature_split, num_trees, depth):
        super().__init__(error_function, depth)
        self.feature_split = feature_split
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

            self.hypotheses.append(root)
            self.hypotheses_votes.append(vote)

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

    def _build_tree(self, examples, attributes, labels):
        self.root = self._ID3(examples, attributes, labels, self.depth)
        return self.root

    def _test(self, root, S):
        preds = np.zeros(len(S))
        for i, s in enumerate(S):
            preds[i] = self._prediction(root, s)
        return preds

    def _cast_vote(self, root, examples, labels):
        preds = self._test(root, examples)

        error = self._test_error(preds, labels)
        vote = np.log2(1 - error/2*error)

        return vote

    def _prediction(self, root, example):
        while root.children:
            attribute = example[root.attribute]
            if attribute in root.children:
                root = root.children[attribute]

        return root.label

    def _ID3(self, S, attributes, labels, depth):
        dom_label = self._dominant_label(labels)

        if len(set(labels)) == 1 or not attributes or depth == 0:
            leaf = Node(None)
            leaf.label = dom_label
            return leaf

        if len(attributes) > 1:
            G = {}
            g = random.sample(list(attributes), self.feature_split)
            for attr in g:
                G[attr] = attributes[attr]
            split_attr = self._information_gain(S, G, labels)
        else:
            split_attr = self._information_gain(S, attributes, labels)


        root = Node(split_attr)

        for v in attributes[split_attr]:
            new_branch = Node(v)

            Sv = [sv for i, sv in enumerate(S) if S[i][split_attr] == v]
            Sv_labels = [label for i, label in enumerate(labels) if S[i][split_attr] == v]

            if not Sv:
                new_branch.label = dom_label
                root.add_child(v, new_branch)
            else:
                copy_attr = deepcopy(attributes)
                copy_attr.pop(split_attr)

                root.add_child(v, self._ID3(Sv, copy_attr, Sv_labels, depth - 1))

        return root
        

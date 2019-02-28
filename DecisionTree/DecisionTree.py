from collections import Counter
import numpy as np
from math import log
from copy import deepcopy


class Node():
    '''
        Node that represents a branch in the tree
        params:
            attribute: attribute which this branch takes, options: (None, str)
        properties:
            attribute:
            children: Children of this node
            label: label which this node classifies. options: (None, str)
    '''
    def __init__(self, attribute):
        self.attribute = attribute
        self.children = {}
        self.label = None

    '''
        Adds child to this node
        params:
            v: v symbolizes an instance that an attribute can take
            node: node to add to the children
    '''
    def add_child(self, v, node):
        self.children[v] = node

class DecisionTree():
    '''
        Decision Tree created using ID3 algorithm
        params:
            error_function: The function which to choose the 
                splits of attributes in the data, options: (function)
            depth: depth in which the tree should take, options: (int)

    '''
    def __init__(self, error_function, depth):
        self.error_function = error_function
        self.depth = depth

    '''
        Creates a decision tree using the ID3 algorithm on the examples S
        params:
            S: examples to base the tree from, options: (list of maps, where maps contain keys with each attribute)
            attributes: list of attributes in the training data, 
                options: (map: keys = attribute, val = value an attribute can take)
            labels: labels that correspond to the examples in S, options: (list)
            depth: amount of nodes down a tree can go, options: (int)
    '''
    def _ID3(self, S, attributes, labels, depth, weights= None):
        dom_label = self._dominant_label(labels)

        if len(set(labels)) == 1 or not attributes or depth == 0:
            leaf = Node(None)
            leaf.label = dom_label
            return leaf
        

        split_attr = self._information_gain(S, attributes, labels, weights)

        root = Node(split_attr)

        for v in attributes[split_attr]:
            new_branch = Node(v)

            Sv = [sv for i, sv in enumerate(S) if S[i][split_attr] == v]
            Sv_labels = [label for i, label in enumerate(labels) if S[i][split_attr] == v]

            if weights is not None:
                Sv_weights = [weight for i, weight in enumerate(weights) if S[i][split_attr] == v]

            if not Sv:
                new_branch.label = dom_label
                root.add_child(v, new_branch)
            else:
                copy_attr = deepcopy(attributes)
                copy_attr.pop(split_attr)

                if weights is not None:
                    root.add_child(v, self._ID3(Sv, copy_attr, Sv_labels, depth - 1, Sv_weights))
                else:
                    root.add_child(v, self._ID3(Sv, copy_attr, Sv_labels, depth - 1))

        return root

    '''
        Calls the private method ID3 to create a root node for the tree
        params:
            S: examples to base the tree from, options: (list of maps, where maps contain keys with each attribute)
            attributes: list of attributes in the training data, 
                options: (map: keys = attribute, val = value an attribute can take)
            labels: labels that correspond to the examples in S, options: (list)
    '''
    def create_tree(self, S, attributes, labels, weights= None):
        self.root = self._ID3(S, attributes, labels, self.depth, weights=weights)

    '''
        Gets the dominant label from the list
        params:
            list_: List with labels
    '''
    def _dominant_label(self, list_):
        count = Counter(list_)
        return count.most_common(1)[0][0]

    '''
        Calculates the information gain of splitting the attributes
        params:
            S: set of examples
            attributes: set of attributes to base the split on
            labels: labels corresponding to set of examples
    '''
    def _information_gain(self, S, attributes, labels, weights= None):
        total_error = self.error_function(labels)
        gain = -1
        split_attr = None

        for attr in attributes:
            gain_attr = total_error
            for v in attributes[attr]:
                if weights is not None:
                    Sv_labels = [(label, weight) for i, (label, weight) in enumerate(zip(labels, weights)) if S[i][attr] == v]
                    Sv_weights = [label[1] for label in Sv_labels]
                else:
                    Sv_labels = [label for i, label in enumerate(labels) if S[i][attr] == v]

                if Sv_labels:
                    gain_attr -= (len(Sv_labels)/len(labels)) * self.error_function(Sv_labels) if weights is None else (sum(Sv_weights)/sum(weights)) * self.error_function(Sv_labels)

            if gain_attr > gain:
                gain = gain_attr
                split_attr = attr

        return split_attr

    '''
        Tests the model on a set of examples
        params:
            S: set of examples to test the model on
    '''
    def test(self, S):
        predicted_labels = []
        for s in S:
            predicted_labels.append(self._prediction(s))
        return predicted_labels

    '''
        Gives a prediction based on the tree given the example.
        params:
            example: Example to be given a prediction
    '''
    def _prediction(self, example):
        root = self.root

        while root.children:
            attribute = example[root.attribute]
            if attribute in root.children:
                root = root.children[attribute]

        return root.label

    '''
        Calculates the accuracy given the predicted labels and expected labels
        params:
            predicted_labels: labels predicted by the model
            expected_labels: ground truth labels
    '''
    def test_error(self, predicted_labels, expected_labels):
        count = 0
        for pl, el in zip(predicted_labels, expected_labels):
            if pl == el:
                count += 1
        return 1 - count/len(expected_labels)
        

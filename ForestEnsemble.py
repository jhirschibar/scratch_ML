#!/usr/bin/python3

import csv
import numpy as np 
import ast
from datetime import datetime
from math import log, floor, ceil

class Utility(object):
    
    # This method computes entropy for information gain
    def entropy(self, class_y):
        # Input:            
        #   class_y         : list of class labels (0's and 1's)
        #
        # Example:
        #    entropy([0,0,0,1,1,1,1,1,1]) = 0.92

        entropy = 0
        #print('entropy, current class y:' ,class_y)
        if len(class_y)==0:
            entropy = 1
            return entropy
        count_0 = class_y.count(0)
        count_1 = class_y.count(1)
        p0 = count_0 / (count_1 + count_0)
        p1 = count_1 / (count_1 + count_0)
        if p0==0 or p1==0: #when all samples are of the same class
            entropy = 0
        else:
            entropy = (-1)*p0 *log(p0, 2) - p1*(log(p1, 2))
        return entropy
    
    def partition_classes(self, X, y, split_attribute, split_val):
        # Inputs:
        #   X               : data containing all attributes
        #   y               : labels
        #   split_attribute : column index of the attribute to split on
        #   split_val       : a numerical value to divide the split_attribute
        '''
            Example:
            X = [[3, 10],                 y = [1,
                [1, 22],                      1,
                [2, 28],                      0,
                [5, 32],                      0,
                [4, 32]]                      1]
            Here, columns 0 and 1 represent numeric attributes.

            Consider the case where we call the function with split_attribute = 0 and split_val = 3 (mean of column 0)
            Then we divide X into two lists - X_left, where column 0 is <= 3  and X_right, where column 0 is > 3.

            X_left = [[3, 10],                 y_left = [1,
                    [1, 22],                           1,
                    [2, 28]]                           0]

            X_right = [[5, 32],                y_right = [0,
                    [4, 32]]                           1]
            ''' 

            left_index = np.where(X[:,split_attribute]<split_val)[0]
            right_index = np.where(X[:,split_attribute]>=split_val)[0]
            X_left = X[left_index]
            y_left = y[left_index]
            X_right = X[right_index]
            y_right = y[right_index]
            return (X_left, X_right, y_left, y_right)

    def information_gain(self, previous_y, current_y):
        # Inputs:
        #   previous_y: the distribution of original labels (0's and 1's)
        #   current_y:  the distribution of labels after splitting based on a particular
        #               split attribute and split value
        """
        Example:
        previous_y = [0,0,0,1,1,1]
        current_y = [[0,0], [1,1,1,0]]
        info_gain = 0.45915
        """
        info_gain = 0
        H = self.entropy(list(previous_y))
        H0 = self.entropy(list(current_y[0]))
        H1 = self.entropy(list(current_y[1]))
        p0 = len(current_y[0])/len(previous_y)
        p1 = len(current_y[1])/len(previous_y)
        info_gain = H - ((H0 * p0) + (H1*p1))
        return info_gain

    def best_split(self, X, y):
        # Inputs:
        #   X: Data containing all attributes
        #   y: labels
                split_attribute = 0
        split_value = 0
        X_left, X_right, y_left, y_right = [], [], [], []

        if type(X)==list:
            X = np.array(X)
        if type(y) ==list:
            y = np.array(y)
        d = X.shape[1]
        m = np.random.randint(0, d)
        m = list(np.random.choice(range(d), m, replace=False))
        info_gain = 0
        for split_attribute in m:
            split_value = np.mean(X[:,split_attribute])
            temp_x_left, temp_x_right, temp_y_left, temp_y_right = self.partition_classes(X, y, split_attribute, split_value)
            current_y = [temp_y_left, temp_y_right]
            temp_info_gain = self.information_gain(y, current_y)
            if temp_info_gain >= info_gain:
                info_gain = temp_info_gain
                X_left, X_right, y_left, y_right = temp_x_left, temp_x_right, temp_y_left, temp_y_right
        return split_attribute, split_value, X_left, X_right, y_left, y_right
    
class DecisionTree(object):
    def __init__(self, max_depth):
        # Initializing the tree as an empty dictionary
        self.tree = []
        self.max_depth = max_depth
        

    def learn(self, X, y, par_node = {}, depth=0):    
        util=Utility()
        self.tree.append(par_node)
        for depth in range(self.max_depth):
            depth_counter = 0
            for par_node in self.tree:
                if par_node['depth']==depth and par_node['type']=='leaf':
                    try:
                        split_attribute, split_value, X_left, X_right, y_left, y_right = util.best_split(par_node['data_subset'], par_node['labels'])
                    except IndexError as e:
                        print('depth:', depth)
                        print('y', node['labels'])
                    if len(X_left)<=2 or len(X_right)<=2 or len(y_left)<=2 or len(y_right)<=2:
                        pass #this node is a leaf
                    else:
                        par_node['type']='parent'
                        par_node['split_attribute']=split_attribute
                        par_node['split_value']=split_value
                        par_node['depth_id'] = depth_counter

                        left_node, right_node = {}, {}
                        left_node['depth']=depth + 1
                        left_node['type']='leaf'
                        left_node['side']='left'
                        left_node['data_subset']=X_left
                        left_node['labels']=y_left
                        left_node['parent_depth_id']=depth_counter
                        self.tree.append(left_node)

                        right_node['depth']=depth + 1
                        right_node['type']='leaf'
                        right_node['side']='right'
                        right_node['data_subset']=X_right
                        right_node['labels']=y_right
                        right_node['parent_depth_id']=depth_counter
                        self.tree.append(right_node)
                        depth_counter+=1
        return    

    def classify(self, record): #threshold for the decision variable can be tuned
        label = 'None'
        side = 'left'
        parent_id = 'None'

        for depth in range(self.max_depth+1):
            nodes = [x for x in self.tree if x['depth']==depth]
            
            if len(nodes)>1:
                nodes = [x for x in nodes if x['parent_depth_id']==parent_id and x['side']==side]
            
            node = nodes[0]
            node_type = node['type']
            if node_type != 'leaf':
                split_attribute = node['split_attribute']
                split_value = node['split_value']
                parent_id = node['depth_id']
                if record[split_attribute] >= split_value:
                    side='right'
                else:
                    side='left'
            else:
                leaf_y = node['labels']
                eval = sum(leaf_y)/len(leaf_y)
                if eval > .5: #this threshold can be tuned as needed
                    label = 1
                else:
                    label = 0
                return label

class RandomForest(object):
    num_trees = 0
    decision_trees = []
    # bootstraps_datasets is a list of lists, where each list in bootstraps_datasets is a bootstrapped dataset.
    bootstraps_datasets = []

    # the true class labels, corresponding to records in the bootstrapping datasets
    # bootstraps_labels is a list of lists, where the 'i'th list contains the labels corresponding to records in
    # the 'i'th bootstrapped dataset.
    bootstraps_labels = []

    def __init__(self, num_trees): 
        self.num_trees = num_trees
        self.decision_trees = [DecisionTree(max_depth=12) for i in range(num_trees)]
        self.bootstraps_datasets = []
        self.bootstraps_labels = []

    def _bootstrapping(self, XX, n): #XX is the starting dataset
        samples = [] # sampled dataset
        labels = []  # class labels for the sampled records
        if type(XX) == list:
            XX = np.array(XX)
        indexes = np.random.choice(n, n, replace=True)
        samples = XX[indexes, :-1] #X values
        labels = XX[indexes, -1] #y values
        return (samples, labels)

    def bootstrapping(self, XX):
        # Initializing the bootstap datasets for each tree
        for i in range(self.num_trees):
            data_sample, data_label = self._bootstrapping(XX, len(XX))
            self.bootstraps_datasets.append(data_sample)
            self.bootstraps_labels.append(data_label)

    def fitting(self):
        # Train `num_trees` decision trees using the bootstraps datasets and labels by calling the learn function from DecisionTree class.
        for i in range(self.num_trees):
            X = self.bootstraps_datasets[i]
            y = self.bootstraps_labels[i]
            #self.decision_trees[i].iterate_through_nodes(X, y)
            par_node = {'depth':0, 'type':'leaf', 'data_subset':X, 'labels':y, 'parent_depth_id':'None'}
            self.decision_trees[i].learn(X,y, par_node = par_node)

    def voting(self, X): #where each tree casts its "vote" 
        y = []
        counter = 0
        for record in X:
            votes = []
            for i in range(len(self.bootstraps_datasets)):
                dataset = self.bootstraps_datasets[i].tolist()
                
                if record not in dataset:
                    OOB_tree = self.decision_trees[i]
                    effective_vote = OOB_tree.classify(record)
                    votes.append(effective_vote)

            counts = np.bincount(votes)
            if len(counts) == 0:
                # Special case: Handle the case where the record is not an out-of-bag sample for any of the trees.
                effective_vote = np.random.randint(0,1)
                counter +=1
                y = np.append(y, effective_vote)
                #############################################
            else:
                y = np.append(y, np.argmax(counts))
        print('number that were not OOB:', counter)        
        return y


def main(forest_size=200, path=None): #path = path to dataset
    # start time 
    start = datetime.now()
    X = list()
    y = list()
    XX = list()  # Contains data features and data labels
    numerical_cols = set([i for i in range(0, 9)])  # indices of numeric attributes (columns)

    # Loading data set
    print("reading the data")
    with open(path) as f:
        next(f, None)
        for line in csv.reader(f, delimiter=","):
            xline = []
            for i in range(len(line)):
                if i in numerical_cols:
                    xline.append(ast.literal_eval(line[i]))
                else:
                    xline.append(line[i])
            X.append(xline[:-1])
            y.append(xline[-1])
            XX.append(xline[:])

    # Initializing a random forest.
    randomForest = RandomForest(forest_size)

    # Creating the bootstrapping datasets
    print("creating the bootstrap datasets")
    randomForest.bootstrapping(XX)

    # Building trees in the forest
    print("fitting the forest")
    randomForest.fitting()

    # Calculating an unbiased error estimation of the random forest
    # based on out-of-bag (OOB) error estimate.
    y_predicted = randomForest.voting(X)

    # Comparing predicted and true labels
    results = [prediction == truth for prediction, truth in zip(y_predicted, y)]

    # Accuracy
    accuracy = float(results.count(True)) / float(len(results))

    print("accuracy: %.4f" % accuracy)
    print("OOB estimate: %.4f" % (1 - accuracy))

    # end time
    print("Execution time: " + str(datetime.now() - start))

if __name__=='__main__':
    main()
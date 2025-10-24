import numpy as np
from collections import Counter


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None,*,value=None):
        self.feature = feature  #what do we split by
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value= value


    def is_leaf_node():
        return self.value is not None    #if value exists it is leaf node (checking)


class DecisionTree:
    def __init__(self, min_sample_split=2, max_depth=100, n_features=None):
        self.min_samples_split= min_sample_split          #Minimum number of samples required to split a node. If fewer than this, the node becomes a leaf.
        self.max_depth= max_depth                     #Maximum depth (levels) of the tree. Limits how large the tree grows (prevents overfitting).
        self.n_features= n_features                  #number of features, add randomness to the tree (used in RF)
        self.root=None


    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1],self.n_features) #No error
        self.root = self._grow_tree(X, y)
    
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))            #length of amount of unique values of y will be the labels
        
        #check the stopping criteria
        if (depth>=self.max_depth or n_labels==1 or n_samples<self.min_samples_split):
            leaf_value = self._most_common_label(y)                 #helper function
            return Node(value=leaf_value)
        
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)

        #find the best split (entropy)
        best_feature, best_thresh = self._best_split(X, y, feat_idxs)                            #considering create a new split

        #create child nodes. increase depth by 1
        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)                   #split calculates what left and right should be
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        return Node(best_feat, best_thresh, left, right)


    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None


        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for thr in thresholds:
                #calculate information gain
                gain = self._information_gain(y, X_column, thr)

                if gain> best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr
        return split_idx, split_threshold


    def _information_gain(self, y, X_column, thr):
        #parent entropy
        parent_entropy = self._entropy(y)

        #create children
        left_idx, right_idx = self._split(X_column, threshold)
        if len(left_idx) == 0 or len(right_idx) == 0:
            return 0

        #calculate weighted entropy of children
        n = len(y)
        n_l, n_r = len(left_idx), len(right_idx)
        e_l, e_r = self._entropy(y[left_idx]), self._entropy(y[right_idx])
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r

        #calculate the IG
        information_gain = parent_entropy - child_entropy
        return information_gain

    def _split(self, X_column, split_thresh):
        left_idxs=np.argwhere(X_column <= split_thresh).flatten()                 #sets up boundaries to split X value
        right_idxs=np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs    

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p*np.log(p) for p in ps if p>0])

    def _most_common_label(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value

    def predict(self, X):
        np.arry([self._traverse_tree(x, self.root) for x in X])


    def _traverse_tree(self, x, node):
        if node.is_leaf_node():                 #check if it is the final node
            return node.value
        

        if x[node.feature] <= node.threshold:                   #check which side of threshold it goes on
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
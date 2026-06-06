import numpy as np
import math
from tqdm import tqdm

class GradientBoosterClassifier:
    def __init__(self, n_trees, learning_rate, max_tree_depth):
        #initialize parameters
        self.n_trees = n_trees
        self.learning_rate = learning_rate
        self.max_tree_depth = max_tree_depth

        self.trees = [] #the list that will contain all of the trees that were made
        self.F_0 = None #initialize the model's "first" guess

    def _sigmoid(self, F):
        """
        Turns log-odds into a probability value (takes whatever F is and turns that into a 
        probability of some class occuring)
        """
        return 1/(1 + np.exp(-F))

    def fit(self, X, y):
        """Fit the model to the training data.
        Args:
            X: array-like of shape (n_samples, n_features)
            y: array-like of shape (n_samples,)
        """
        p_mean = np.mean(y) #the average "baseline" guess to begin with
        self.F_0 = math.log(p_mean/(1-p_mean)) #convert mean value into log-odds form
        
        current_F = np.full(X.shape[0], self.F_0) #model's current running score for each training sample

        for m in tqdm(range(self.n_trees), desc = "Training trees"):
            probabilities = self._sigmoid(current_F) #compute the pobabilities of our guesses
            residuals = y - probabilities #compute the residuals ("how far off are we")
            
            tree = RegressionTree(max_depth = self.max_tree_depth, min_samples_split = 2) #make a new regression tree
            tree.fit(X, residuals) #fit the tree to the residuals (the tree is learning to predict the residuals)

            leaf_ids = tree.apply(X) #where each training sample ends up in the tree (which leaf [id] does it fall into)
            weights = probabilities * (1 - probabilities) #computes the "weights" for the samples (how much each sample should influence the tree's predictions)

            for leaf in np.unique(leaf_ids):
                mask = (leaf_ids == leaf) #pick out only the training samples that fall into this leaf
                gamma = residuals[mask].sum() / weights[mask].sum() #the number that the leaf should predict when a sample falls into that leaf during prediction time
                tree._update_leaf(leaf, gamma) #update the leaf value to be the optimal gamma

            #scale the tree's contribution by the learning rate and add it to the running scores
            current_F += self.learning_rate * tree.predict(X)

            self.trees.append(tree) #add the tree to the list of trees

    def predict(self, X, threshold=0.5):
        """Predict class labels for samples in X.
        Args:
            X: array-like of shape (n_samples, n_features)
            threshold: float, default=0.5
                Threshold for converting predicted probabilities to class labels.
        Returns:
            The predicted class labels
        """
        return (self.predict_proba(X) >= threshold).astype(int)

    def predict_proba(self, X):
        """Predict class probabilities for samples in X.
        Args:
            X: array-like of shape (n_samples, n_features)
        Returns:
            The predicted class probabilities
        """
        F = np.full(X.shape[0], self.F_0) #start with the initial guess
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X) #add the contribution of each tree
        return self._sigmoid(F) #convert log-odds to probabilities

class RegressionTree:
    def __init__(self, max_depth, min_samples_split):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root_node = None
        
        self.leaf_counter = 0 #keeps track of the number of leaf nodes created (to distinguish them)
        self.leaves = {}

    def fit(self, X, y): 
        """
        Fits a regression Tree to the data and returns the RegressionTree.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        self.leaf_counter = 0
        self.leaves = {}
        self.root_node = self._build_tree(X, y, depth=0)

        return self


    def predict(self, X):
        """
        Returns the predicted values for an inputted array of features X.
        Args:
            X (np.array): the features of a bunch of data points
        Returns:
            (np.array): an array of a predicted values based on 'walking the tree'
        """
        #goes through all rows in X and walks the decision tree with it. Returns the value of the leaf node and adds it to the array
        return np.array([self._walk_tree(x, self.root_node).value for x in X]) 

    def apply(self, X):
        """
        Returns the id of the leaf node that a set of features "landed" in (does this for an array of X features)
        Args:
            X (np.array): the features of a bunch of data points
        Returns:
            (np.array): an array of leaf node ids based on 'walking the tree'
        """
        #goes through all rows in X and walks the decision tree with it. Returns the id of the leaf node and adds it to the array
        return np.array([self._walk_tree(x, self.root_node).id for x in X]) 

    def squared_error(self, y):
        """Measures how "spread out" the target values are based on the following formula: Σ (yᵢ - ȳ)²
        Args:
            y: the target features
        Returns:
            float: how spread out the target values are from each other
        """
        if len(y) == 0: #guard for empty arrays
            return 0.0
        
        avg_val = y.mean()
        difference = [y[i] - avg_val for i in range(len(y))]
        sqrd = [difference[i] * difference[i] for i in range(len(difference))]
        summation = sum(sqrd)

        return summation

    def _select_best_split(self, X, y):
        """
        Selects the best split based on which subnodes produce the smallest mean squared error.
        Returns the feature index and the threshold that determines which values go to which node.
        """
        n_samples, n_features = X.shape
        best_feature = None
        best_threshold = None
        best_error = float('inf')

        for feature in range(n_features):
            feature_values = np.unique(X[:,feature])

            for i in range(len(feature_values)-1):
                threshold = (feature_values[i] + feature_values[i+1])/2

                left_mask = X[:, feature] < threshold
                right_mask = X[:, feature] >= threshold
                left_y = y[left_mask]
                right_y = y[right_mask]

                error = (np.sum((left_y - np.mean(left_y)) ** 2) + np.sum((right_y - np.mean(right_y)) ** 2))

                if error < best_error:
                    best_error = min(error, best_error)
                    best_threshold = threshold
                    best_feature = feature

        return best_feature, best_threshold

    def _make_node(self, X, y, index):
        """Makes a node (either a leaf or a decision node) based on the inputted index (if index is None, make a leaf; otherwise make a decision node)
        Args:
            X: the features of the data points that have made it to this node so far
            y: the target values of the data points that have made it to this node so far
            index: the index of the feature and threshold to split on (if None, make a leaf)
        Returns:             
            Node: the new node that was made
        """
        new_node = Node()
        if index is not None: #meaning that we want to make a node (not a leaf) since we have a best_feature and best_threshold
            new_node.feature = index[0]
            new_node.threshold = index[1]
            #set left_node and right_node later in _build_tree
        else: #we want to build a leaf node
            new_node.value = y.mean()
            new_node.id = self.leaf_counter
            self.leaf_counter += 1 #increment leaf counter since we have made a leaf
            self.leaves[new_node.id] = new_node #append this newly created leaf to the dictionary of leaves (with the key being the leaf's id and the value being the leaf node itself)
        
        return new_node


    def _build_tree(self, X, y, depth): 
        """
        Recursively build a regression tree.
        """
        # Stop if depth reaches set max depth, all values are the same, or we are below the minimum number of samples per node.
        if len(y) == 0 or depth >= self.max_depth or np.all(y == y[0]) or len(y) < self.min_samples_split:
            return self._make_node(X, y, None)

        feature_index, threshold = self._select_best_split(X, y)

        if feature_index is None:
            return self._make_node(X, y, None)
        
        left_mask = X[:, feature_index] < threshold
        right_mask = ~left_mask

        # Recursively build the children using the feature index and threshold from splitting.
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        node = self._make_node(X, y, (feature_index, threshold))
        node.left_node = left_child
        node.right_node = right_child
        return node
    
    def _walk_tree(self, X, node):
        """Walks down the tree till it gets to a leaf node and returns that leaf
        Args:
            X: a single row of features
            node: the root node of the tree to walk down
        Returns:
            node: the leaf node that best reflects the inputted features
        """
        current_node = node
        while not current_node.is_leaf():
            #compare the feature value to the node's threshold
            feature_value = X[current_node.feature]
            if feature_value < current_node.threshold:
                #go left
                current_node = current_node.left_node
            else:
                #go right
                current_node = current_node.right_node

        return current_node
    
    def _update_leaf(self, leaf_id, gamma):
        self.leaves[leaf_id].value = gamma
    

class Node:
    """
    Class that represents a node on the regression tree. Can be either a node (it "asks" a question about a particular feature to 
    narrow down the prediction) or it is a leaf (it gives a prediction)"""
    def __init__(self):
        #node variables
        self.feature = None #index of the column/feature to test
        self.threshold = None #cuttoff value
        self.left_node = None #points to the node to the right (child for yes) --> like a linked list
        self.right_node = None #points to the node to the left (child for no) --> like a linked list

        #leaf variables
        self.value = None #the prediction value
        self.id = None #label for the booster

    def is_leaf(self):
        """Is this node a leaf or not (only leaves has self.value as not None)"""
        return self.value is not None
    
    

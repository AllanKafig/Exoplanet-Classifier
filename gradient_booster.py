import numpy as np
import math

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

        for m in range(self.n_trees):
            probabilities = self._sigmoid(current_F) #compute the pobabilities of our guesses
            residuals = y - probabilities #compute the residuals ("how far off are we")
            
            tree = RegressionTree(max_depth = self.max_tree_depth, min_samples_split = 2) #make a new regression tree
            tree.fit(X, residuals) #fit the tree to the residuals (the tree is learning to predict the residuals)

            leaf_ids = tree.apply(X) #where each training sample ends up in the tree (which leaf [id] does it fall into)
            weights = probabilities * (1 - probabilities) #computes the "weights" for the samples (how much each sample should influence the tree's predictions)

            for leaf in np.unique(leaf_ids):
                mask = (leaf_ids == leaf) #pick out only the training samples that fall into this leaf
                gamma = residuals[mask].sum() / weights[mask].sum() #the number that the leaf should predict when a sample falls into that leaf during prediction time
                tree.update_leaf(leaf, gamma) #update the leaf value to be the optimal gamma

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
    
class TreeNode:
    def __init__(self, left_child, right_child, feature_index, threshold):
        self.left_child = left_child
        self.right_child = right_child
        self.feature_index = feature_index
        self.threshold = threshold

class RegressionTree:
    def __init__(self, max_depth, min_samples_split):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root_node = None

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def apply(self, X):
        pass

    def select_best_split(self, X, y):
        n_samples, n_features = X.shape
        best_feature = None
        best_threshold = None
        best_error = float('inf')

        for feature in range(n_features):
            feature_values = np.unique(X[:, feature])

            for i in range(len(feature_values) - 1):
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

    def make_leaf(self, y):
        """
        Under Construction
        """
        leaf = TreeNode(
            None,
            None,
            None,
            np.mean(y)
        )

    def build_tree(self, X, y, depth):
        """
        Recursively build a regression tree.
        """
        if depth >= self.max_depth or np.all(y == y[0]) or len(y) < self.min_samples_split:
            return self.make_leaf(X, y)

        feature_index, threshold = self.select_best_split(X, y)

        if feature_index is None:
            return self.make_leaf(X, y)
        
        left_mask = X[:, feature_index] < threshold
        right_mask = ~left_mask

        left_child = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self.build_tree(X[right_mask], y[right_mask], depth + 1)

        return TreeNode(left_child, right_child, feature_index, threshold)


    
    

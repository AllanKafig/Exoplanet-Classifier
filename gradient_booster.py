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

    def _build_tree(self, X, y, depth):
        pass
    
    

from .Baseclass import Model
from graphviz import Digraph
import numpy as np
from numba import njit

@njit
def variance_jit(y):
    mean = np.mean(y)
    total = 0.0
    for i in y:
        total += (i - mean) ** 2
    return total / len(y)



class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        """
        Input: min_sample_split int, max_depth int or None
        Output: None
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.prediction = value
        self.is_leaf = value is not None

    def __repr__(self):
        if self.is_leaf:
            return f"Leaf(value={self.prediction:.2f})"
        return f"Node(feature={self.feature}, threshold={self.threshold:.2f})"

    def print_tree(self, depth=0):
        indent = "  " * depth
        if self.is_leaf:
            print(f"{indent}Predict: {self.prediction:.2f}")
        else:
            print(f"{indent}If feature[{self.feature}] <= {self.threshold}:")
            self.left.print_tree(depth + 1)
            print(f"{indent}else:")
            self.right.print_tree(depth + 1)
            
    def to_dict(self):
        if self.is_leaf:
            return {'value': self.prediction}
        return {
            'feature': self.feature,
            'threshold': self.threshold,
            'left': self.left.to_dict(),
            'right': self.right.to_dict()
        }


class DecisionTree(Model):

    def __init__(self, min_sample_split = 2,max_depth = None):
        """
        Input: min_sample_split int, max_depth int or None
        Output: None
        """
        super().__init__(name="DecisionTree")
        self.min_sample_split = min_sample_split
        self.root = None
        self.max_depth = max_depth
        
    def _predict_one(self, x, node):
        """
        Input: x 1D array (single sample), node (Node)
        Output: Predicted value float
        """
        if node.is_leaf:
            return node.prediction
        else:
            if x[node.feature] <= node.threshold:
                return self._predict_one(x, node.left)
            else:
                return self._predict_one(x, node.right)
            
    def predict(self, X):
        """
        Input: X 2D array of input samples
        Output: 1D array of predicted values
        """
        return np.array([self._predict_one(x, self.root) for x in X])
    
    def fit(self, X, y):
        """
        Input: X 2D array of features, y 1D array of targets
        Output: None
        """
        # Ensure numeric types for Numba-compiled functions
        self.global_mean = y.mean() 
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.root = self._growtree(X, y)

    def _split(self,X,y,feature_index, threshold):
        """
        Args: X A 2D array of input samples.
            y feature var array
            feature index the feature on which split occurs int
            threshold the value on which the data is split
        """
        mask = X[:, feature_index] <= threshold
        return X[mask], X[~mask], y[mask], y[~mask]
    def _variance(self, y):
        return variance_jit(y)
    
    def _bestsplit(self, X, y):
        """
        Input: X 2D array of features, y 1D array of targets
        Output: Tuple of (best_feature_index, best_threshold)
        """
        n = len(y)
        best_score = float('inf')
        best_feature_index = None
        best_threshold = None

        for i in range(X.shape[1]):
            x_column = X[:, i]
            sorted_indices = np.argsort(x_column)
            x_sorted = x_column[sorted_indices]
            y_sorted = y[sorted_indices]

            for j in range(1, len(y)):
                if x_sorted[j] == x_sorted[j - 1]:
                    continue
                threshold = (x_sorted[j] + x_sorted[j - 1]) / 2

                y_left = y_sorted[:j]
                y_right = y_sorted[j:]

                n_left = len(y_left)
                n_right = len(y_right)

                var_left = np.var(y_left)
                var_right = np.var(y_right)

                total_var = (n_left / n) * var_left + (n_right / n) * var_right

                if total_var < best_score:
                    best_score = total_var
                    best_threshold = threshold
                    best_feature_index = i

        return best_feature_index, best_threshold
    def _growtree(self, X, y, depth=0):
        """
        Input: X 2D array of features, y 1D array of targets, depth int
        Output: Root Node of a (sub)tree, or leaf node
        """
        # If no split was found, return a leaf
            # If there are no samples left, return a leaf with the global mean
        if y.size == 0:
            return Node(value=self.global_mean)
        # stopping condition: max depth reached or pure leaf or not enough samples
        n_samples = len(y)
        if ((self.max_depth is not None and depth >= self.max_depth) or 
            n_samples < self.min_sample_split or
            np.all(y == y[0])):
            return Node(value=y.mean()) 
        feature_index, threshold = self._bestsplit(X,y)
        if feature_index is None or threshold is None:
            return Node(value=y.mean())
        X_left, X_right, y_left, y_right = self._split(X,y,feature_index,threshold)
        # recursively build left and right subtrees
        right_subtree = self._growtree(X_right,y_right,depth+1)
        left_subtree = self._growtree(X_left,y_left,depth+1)
        return Node(feature=feature_index, threshold=threshold,
            left=left_subtree, right=right_subtree)

    def export_graphviz(self, out_file='tree', format='png'):
        """
        Input: out_file str (filename without extension), format str (png, pdf, etc.)
        Output: Graphviz render saved to disk
        """
        def add_nodes(dot, node, node_id=0):
            if node.is_leaf:
                dot.node(str(node_id), f'Predict: {node.prediction:.2f}', shape='box')
                return node_id
            dot.node(str(node_id), f'X[{node.feature}] <= {node.threshold}')
            left_id = node_id + 1
            left_id = add_nodes(dot, node.left, left_id)
            right_id = left_id + 1
            right_id = add_nodes(dot, node.right, right_id)
            dot.edge(str(node_id), str(left_id), label='True')
            dot.edge(str(node_id), str(right_id), label='False')
            return right_id
        
        dot = Digraph()
        add_nodes(dot, self.root)
        dot.render(out_file, format=format, cleanup=True)
from graphviz import Digraph
import numpy as np

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
            print(f"{indent}If feature[{self.feature}] <= {self.threshold:.2f}:")
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


class DecisionTree:

    def __init__(self, min_sample_split = 2,max_depth = None):
        """
        Input: min_sample_split int, max_depth int or None
        Output: None
        """
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
        self.root = self._growtree(X, y)

    def _split(self,X,y,feature_index, threshold):
        """
        Args: X A 2D array of input samples.
            y feature var array
            feature index the feature on which split occurs int
            threshold the value on which the data is split
        """
        # split the dataset into left and right groups based on the threshold
        X_left, X_right = [],[]
        y_left, y_right = [], []
        for x in range(len(X)):
            if X[x,feature_index] <= threshold:
                X_left.append(X[x])
                y_left.append(y[x])
            else:
                X_right.append(X[x])
                y_right.append(y[x])
        return np.array(X_left), np.array(X_right), np.array(y_left), np.array(y_right)
    
    def _variance(self, y):
        """
        Input: 1d array of feature vars
        Output: Var value float
        """
        # calculate the mean squared error for the values
        mean = y.mean()
        total = 0
        for i in y:
            total = total + (i-mean)**2
        return total/len(y)
    
    def _bestsplit(self,X,y):
        """
        Input: X 2D array of features, y 1D array of targets
        Output: Tuple of (best_feature_index, best_threshold)
        """
        # loop through all features and thresholds to find the best split
        n = len(y)
        best_score = float('inf')
        best_threshold = None
        best_feature_index = None
        for i in range(X.shape[1]):
            for threshold in np.unique(X[:,i]):
                X_left, X_right, y_left, y_right = self._split(X,y,i,threshold)
                n_left = len(y_left)
                n_right = len(y_right)
                # skip splits that don't divide the data
                if n_left == 0 or n_right == 0:
                    continue
                total_var = (n_left/n)*(self._variance(y_left)) + (n_right/n)*(self._variance(y_right))
                if total_var < best_score:
                    best_score = total_var
                    best_threshold = threshold
                    best_feature_index = i
        if best_feature_index is not None:
            # optional: refine the best threshold between adjacent unique values
            feature_values = np.unique(X[:, best_feature_index])
            idx = np.where(feature_values == best_threshold)[0][0]
            if idx == 0 or idx == len(feature_values) - 1:
                return best_feature_index, best_threshold
            Z = np.linspace(feature_values[idx-1], feature_values[idx+1],num=20)
            for threshold in Z:
                X_left, X_right, y_left, y_right = self._split(X,y,best_feature_index,threshold)
                n_left = len(y_left)
                n_right = len(y_right)
                if n_left == 0 or n_right == 0:
                    continue          
                total_var = (n_left/n)*(self._variance(y_left)) + (n_right/n)*(self._variance(y_right))
                if total_var < best_score:
                    best_score = total_var
                    best_threshold = threshold
            return best_feature_index, best_threshold
    def _growtree(self, X, y, depth=0):
        """
        Input: X 2D array of features, y 1D array of targets, depth int
        Output: Root Node of a (sub)tree, or leaf node
        """
        # stopping condition: max depth reached or pure leaf or not enough samples
        n_samples = len(y)
        if (depth >= self.max_depth or 
            n_samples < self.min_sample_split or
            np.all(y == y[0])):
            return Node(value=y.mean()) 
        feature_index, threshold = self._bestsplit(X,y)
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
            dot.node(str(node_id), f'X[{node.feature}] <= {node.threshold:.2f}')
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
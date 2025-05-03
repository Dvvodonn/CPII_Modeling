import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.decision_tree import DecisionTree

# Sample data
X = np.array([
    [800, 2],
    [950, 2],
    [1200, 3],
    [1500, 4]
])
y = np.array([200000, 220000, 300000, 350000])

# Train and predict
tree = DecisionTree(max_depth=2)
tree.fit(X, y)
preds = tree.predict(X)

# Output

# Save visualization
tree.export_graphviz(out_file="outputs/tree", format="png")

# Open image (macOS only)
os.system("open outputs/tree.png")
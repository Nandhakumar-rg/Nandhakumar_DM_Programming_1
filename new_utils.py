"""
   Use to create your own functions for reuse 
   across the assignment

   Inside part_1_template_solution.py, 
  
     import new_utils
  
    or
  
     import new_utils as nu
"""
import numpy as np

def scale_data(X: np.ndarray) -> np.ndarray:
    """
    Scale the data to have floating point values between 0 and 1.

    Args:
        X (np.ndarray): The input data array to scale.

    Returns:
        np.ndarray: The scaled data array.
    """
    # Ensure the data type is float to avoid integer division
    X = X.astype(np.float64)

    # Scale the data to be between 0 and 1
    X /= 255.0

    return X

def filter_and_modify_7_9s(X, y):
    # Filter to keep only 7s and 9s
    filter_mask = (y == 7) | (y == 9)
    X_filtered = X[filter_mask]
    y_filtered = y[filter_mask]

    # Convert labels: 7 -> 0, 9 -> 1
    y_modified = np.where(y_filtered == 7, 0, 1)

    # Create imbalance by removing 90% of 9s (which are now 1s)
    mask_9s = (y_modified == 1)
    indices_to_keep = np.random.choice(np.where(mask_9s)[0], size=int(sum(mask_9s) * 0.1), replace=False)

    mask_to_remove = np.ones(len(y_modified), dtype=bool)
    mask_to_remove[indices_to_keep] = False
    mask_to_remove &= mask_9s  # Keep all 0s and the reduced set of 1s

    X_imbalanced = X_filtered[~mask_to_remove]
    y_imbalanced = y_modified[~mask_to_remove]

    return X_imbalanced, y_imbalanced


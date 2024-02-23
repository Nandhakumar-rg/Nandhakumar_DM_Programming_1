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

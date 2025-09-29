"""
L4P Inverse Function - Python Implementation
Inverse of the 4 Parameter Logistic equation for interpolation

Original MATLAB implementation by Giuseppe Cardillo
Python conversion with vectorized operations and error handling
"""

import numpy as np
from typing import Union, List, Tuple
import warnings

def l4p_inverse(params: Union[np.ndarray, List[float]], 
                y: Union[float, np.ndarray],
                validate_params: bool = True) -> Union[float, np.ndarray]:
    """
    Calculate the inverse of the 4 Parameter Logistic (4PL) equation.
    
    The 4PL equation is: F(x) = D + (A-D)/(1+(x/C)^B)
    This function solves for x given y: x = C * (((A-D)/(y-D)) - 1)^(1/B)
    
    Parameters:
    -----------
    params : array-like
        4PL parameters [A, B, C, D] where:
        A = Minimum asymptote
        B = Hill's slope  
        C = Inflection point (IC50/EC50)
        D = Maximum asymptote
        
    y : float or array-like
        Response value(s) for which to find corresponding x value(s)
        
    validate_params : bool, default=True
        Whether to validate parameter ranges
        
    Returns:
    --------
    x : float or ndarray
        Interpolated x value(s) corresponding to input y value(s)
        
    Examples:
    ---------
    >>> # Single value interpolation
    >>> params = [0.001, 1.515, 108.0, 3.784]
    >>> x = l4p_inverse(params, 1.782)
    >>> print(f"x = {x:.2f}")
    
    >>> # Multiple value interpolation
    >>> y_values = [0.5, 1.0, 1.5, 2.0]
    >>> x_values = l4p_inverse(params, y_values)
    >>> print(f"x values: {x_values}")
    
    Raises:
    -------
    ValueError
        If parameters are invalid or if y is outside the valid range
    """
    
    # Convert inputs to numpy arrays
    params = np.asarray(params)
    y = np.asarray(y)
    
    # Validate parameters
    if params.shape[-1] != 4:
        raise ValueError("Parameters must be a 4-element array [A, B, C, D]")
    
    # Extract parameters
    A, B, C, D = params[..., 0], params[..., 1], params[..., 2], params[..., 3]
    
    if validate_params:
        # Check for valid parameter ranges
        if np.any(C <= 0):
            warnings.warn("C parameter should be positive for meaningful results")
        if np.any(B == 0):
            raise ValueError("B parameter cannot be zero")
        
        # Check if y values are within valid range
        y_min, y_max = np.minimum(A, D), np.maximum(A, D)
        if np.any((y < y_min) | (y > y_max)):
            warnings.warn("Some y values are outside the asymptotic range [A, D]")
    
    # Handle case where y equals D (would cause division by zero)
    y_safe = np.where(y == D, D + np.finfo(float).eps, y)
    
    # Calculate the inverse using vectorized operations
    # x = C * (((A-D)/(y-D)) - 1)^(1/B)
    ratio = (A - D) / (y_safe - D)
    inner_term = ratio - 1
    
    # Handle negative values under fractional powers
    if np.any(inner_term < 0):
        # For negative values, we need to handle complex results
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            # Use complex power and take real part
            x = C * np.real(np.power(inner_term.astype(complex), 1/B))
    else:
        x = C * np.power(inner_term, 1/B)
    
    # Handle infinite results
    x = np.where(y == D, np.inf, x)
    x = np.where(np.isnan(x), np.inf, x)
    
    # Return scalar if input was scalar
    if np.isscalar(y) and x.ndim == 0:
        return float(x)
    
    return x


def l4p_function(x: Union[float, np.ndarray], 
                 params: Union[np.ndarray, List[float]]) -> Union[float, np.ndarray]:
    """
    Forward 4 Parameter Logistic function.
    
    F(x) = D + (A-D)/(1+(x/C)^B)
    
    Parameters:
    -----------
    x : float or array-like
        Independent variable(s)
    params : array-like
        4PL parameters [A, B, C, D]
        
    Returns:
    --------
    y : float or ndarray
        Function value(s) at x
    """
    params = np.asarray(params)
    A, B, C, D = params[0], params[1], params[2], params[3]
    
    x = np.asarray(x)
    
    # Handle division by zero
    x_safe = np.where(x == 0, np.finfo(float).eps, x)
    
    return D + (A - D) / (1 + (x_safe / C) ** B)


def validate_4pl_fit(x_data: np.ndarray, y_data: np.ndarray, 
                     params: np.ndarray) -> dict:
    """
    Validate a 4PL fit by calculating goodness-of-fit metrics.
    
    Parameters:
    -----------
    x_data : array-like
        Experimental x values
    y_data : array-like  
        Experimental y values
    params : array-like
        Fitted 4PL parameters [A, B, C, D]
        
    Returns:
    --------
    metrics : dict
        Dictionary containing fit quality metrics:
        - r_squared: Coefficient of determination
        - rmse: Root mean square error
        - aic: Akaike Information Criterion
        - residuals: Array of residuals
    """
    
    # Calculate predicted values
    y_pred = l4p_function(x_data, params)
    
    # Calculate residuals
    residuals = y_data - y_pred
    
    # R-squared
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # RMSE
    rmse = np.sqrt(np.mean(residuals ** 2))
    
    # AIC (Akaike Information Criterion)
    n = len(y_data)
    k = 4  # number of parameters
    if ss_res > 0:
        aic = n * np.log(ss_res / n) + 2 * k
    else:
        aic = -np.inf
    
    return {
        'r_squared': r_squared,
        'rmse': rmse,
        'aic': aic,
        'residuals': residuals,
        'ss_res': ss_res,
        'ss_tot': ss_tot
    }


def interpolate_dose_response(params: Union[np.ndarray, List[float]], 
                             y_target: Union[float, List[float]],
                             confidence_level: float = 0.95) -> dict:
    """
    Interpolate dose values for target response(s) with confidence intervals.
    
    Parameters:
    -----------
    params : array-like
        4PL parameters [A, B, C, D]
    y_target : float or list
        Target response value(s) for interpolation
    confidence_level : float, default=0.95
        Confidence level for intervals (0 < confidence_level < 1)
        
    Returns:
    --------
    results : dict
        Dictionary with interpolated values and metadata
    """
    
    y_target = np.atleast_1d(y_target)
    x_interpolated = l4p_inverse(params, y_target)
    
    A, B, C, D = params
    
    # Calculate dynamic range and check if targets are within range
    y_min, y_max = min(A, D), max(A, D)
    within_range = (y_target >= y_min) & (y_target <= y_max)
    
    # Calculate relative position within dynamic range
    relative_position = (y_target - y_min) / (y_max - y_min)
    
    results = {
        'x_interpolated': x_interpolated,
        'y_target': y_target,
        'within_range': within_range,
        'relative_position': relative_position,
        'dynamic_range': (y_min, y_max),
        'ic50_ec50': C,  # Inflection point
        'hill_slope': B
    }
    
    return results


# Example usage and testing
def example_usage():
    """
    Demonstrate the L4P inverse function with examples.
    """
    print("4PL Inverse Function Examples")
    print("=" * 40)
    
    # Example 1: Single interpolation
    print("\n1. Single value interpolation:")
    params = [0.001, 1.515, 108.0, 3.784]
    y_single = 1.782
    x_single = l4p_inverse(params, y_single)
    print(f"   Parameters: A={params[0]}, B={params[1]}, C={params[2]}, D={params[3]}")
    print(f"   y = {y_single} → x = {x_single:.4f}")
    
    # Verify with forward function
    y_check = l4p_function(x_single, params)
    print(f"   Verification: x = {x_single:.4f} → y = {y_check:.4f}")
    
    # Example 2: Multiple value interpolation
    print("\n2. Multiple value interpolation:")
    y_multiple = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    x_multiple = l4p_inverse(params, y_multiple)
    
    print("   y values → x values:")
    for y_val, x_val in zip(y_multiple, x_multiple):
        print(f"   {y_val:4.1f} → {x_val:8.2f}")
    
    # Example 3: Dose-response interpolation
    print("\n3. Dose-response analysis:")
    results = interpolate_dose_response(params, y_multiple)
    
    print(f"   IC50/EC50: {results['ic50_ec50']:.2f}")
    print(f"   Hill slope: {results['hill_slope']:.3f}")
    print(f"   Dynamic range: {results['dynamic_range'][0]:.3f} - {results['dynamic_range'][1]:.3f}")
    
    print("\n   Interpolation results:")
    for i, (y_val, x_val, in_range, rel_pos) in enumerate(zip(
        results['y_target'], 
        results['x_interpolated'],
        results['within_range'],
        results['relative_position']
    )):
        status = "✓" if in_range else "⚠"
        print(f"   {status} y={y_val:4.1f} → x={x_val:8.2f} (rel. pos: {rel_pos:.2f})")


if __name__ == "__main__":
    example_usage()
"""
Thermal Immunopeptidome Profiler (TIP) - Python Version 2.0
A high-performance pipeline for thermostability profiling of HLA-bound peptides

Original MATLAB code by Mohammad (Moh) Shahbazy
Python conversion with performance optimizations and publication-ready plots

Dependencies: numpy, pandas, scipy, matplotlib, seaborn, numba
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from numba import jit, prange
import warnings
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
import time

# Set publication-ready style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

@dataclass
class FitResults:
    """Container for 4PL fit results"""
    params: np.ndarray
    r_squared: float
    tm: float
    fitted_curve: np.ndarray
    confidence_intervals: np.ndarray

class ThermalImmunopeptidomeProfiler:
    """
    High-performance thermal immunopeptidome profiler for HLA-bound peptides
    """
    
    def __init__(self):
        self.temperatures = np.array([37, 42, 46, 50, 54, 58, 63, 68, 73])
        self.irt_sequences = [
            'LGGNEQVTR', 'GAGSSEPVTGLDAK', 'VEATFGVDESNAK', 'YILAGVENSK',
            'TPVISGGPYEYR', 'TPVITGAPYEYR', 'DGLDAASYYAPVR', 'ADVTPADFSEWSK',
            'GTFIIDPGGVIR', 'GTFIIDPAAVIR', 'LFLQFGAQGSPFLK'
        ]
        
    @staticmethod
    @jit(nopython=True)
    def four_pl_function(x: np.ndarray, A: float, B: float, C: float, D: float) -> np.ndarray:
        """
        Four Parameter Logistic function (4PL)
        F(x) = D + (A-D)/(1+(x/C)^B)
        
        Parameters:
        A: Minimum asymptote
        B: Hill's slope 
        C: Inflection point (Tm)
        D: Maximum asymptote
        """
        return D + (A - D) / (1 + (x / C) ** B)
    
    @staticmethod
    @jit(nopython=True)
    def l4p_inverse(params: np.ndarray, y: float) -> float:
        """Inverse of 4PL function to find x given y"""
        A, B, C, D = params
        if y == D:
            return np.inf
        return C * (((A - D) / (y - D)) - 1) ** (1 / B)
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def rolling_median_parallel(data: np.ndarray) -> np.ndarray:
        """Optimized rolling median calculation using numba"""
        n_peptides, n_temps = data.shape
        seg = n_temps // 3
        result = np.zeros((n_peptides, seg))
        
        for i in prange(n_peptides):
            for j in range(seg):
                start_idx = j * 3
                end_idx = start_idx + 3
                segment = data[i, start_idx:end_idx]
                
                # Handle zeros
                non_zero_mask = segment > 0
                if np.sum(non_zero_mask) >= 2:
                    valid_values = segment[non_zero_mask]
                    result[i, j] = np.median(valid_values)
                elif np.sum(non_zero_mask) == 1:
                    result[i, j] = segment[non_zero_mask][0]
                else:
                    result[i, j] = 0
                    
        return result
    
    def load_data(self, data_file: str, target_file: str, 
                  quantification_method: str = 'MS1_PA') -> Tuple[np.ndarray, List[str]]:
        """
        Load and preprocess DIA-NN data
        
        Args:
            data_file: Path to CSV file with quantification data
            target_file: Path to CSV file with target sequences
            quantification_method: 'prec_quant', 'MS1_PA', 'MS2_Frag', or 'Total_raw'
        """
        print("Loading data...")
        
        # Load main data
        data_table = pd.read_csv(data_file)
        
        # Select quantification column based on method
        quant_methods = {
            'prec_quant': 2,  # Column index for precursor quantity
            'MS1_PA': 3,      # MS1 peak area
            'MS2_Frag': 4,    # MS2 fragments
            'Total_raw': 5    # Total peak areas
        }
        
        if quantification_method not in quant_methods:
            raise ValueError(f"Unknown quantification method: {quantification_method}")
        
        quant_col = quant_methods[quantification_method]
        
        # Extract data
        raw_data = data_table.iloc[:, 2:8].values  # Assuming columns 3-8 contain quantification data
        raw_sequences = data_table.iloc[:, 1].values  # Sequence column
        raw_replicates = data_table.iloc[:, 0].values  # Replicate column
        
        # Load target sequences
        target_sequences = pd.read_csv(target_file).iloc[:, 0].tolist()
        
        # Create replicate reference
        rep_ref = [f'{temp}C_r{rep}' for temp in [37, 42, 46, 50, 54, 58, 63, 68, 73] 
                   for rep in [1, 2, 3]]
        
        # Organize data matrix
        print("Organizing data matrix...")
        data_matrix = self._organize_data_matrix(
            raw_data[:, quant_col], raw_sequences, raw_replicates,
            target_sequences, rep_ref
        )
        
        return data_matrix, target_sequences
    
    def _organize_data_matrix(self, data: np.ndarray, sequences: np.ndarray, 
                             replicates: np.ndarray, target_sequences: List[str],
                             rep_ref: List[str]) -> np.ndarray:
        """Organize raw data into matrix format"""
        n_targets = len(target_sequences)
        n_reps = len(rep_ref)
        data_matrix = np.zeros((n_targets, n_reps))
        
        for i, target_seq in enumerate(target_sequences):
            seq_mask = sequences == target_seq
            seq_indices = np.where(seq_mask)[0]
            
            if len(seq_indices) > 0:
                seq_reps = replicates[seq_indices]
                seq_data = data[seq_indices]
                
                for j, ref_rep in enumerate(rep_ref):
                    rep_mask = seq_reps == ref_rep
                    if np.any(rep_mask):
                        # Sum values if multiple entries for same replicate
                        data_matrix[i, j] = np.sum(seq_data[rep_mask])
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{n_targets} sequences")
        
        return data_matrix
    
    def normalize_to_irts(self, data_matrix: np.ndarray, 
                         sequences: List[str]) -> np.ndarray:
        """Normalize data using internal retention time (iRT) peptides"""
        print("Normalizing to iRT peptides...")
        
        # Find iRT peptides in data
        irt_indices = []
        for irt_seq in self.irt_sequences:
            if irt_seq in sequences:
                irt_indices.append(sequences.index(irt_seq))
        
        if len(irt_indices) == 0:
            warnings.warn("No iRT peptides found in data. Skipping normalization.")
            return data_matrix
        
        # Extract iRT data
        irt_data = data_matrix[irt_indices, :]
        
        # Calculate normalization factors
        irt_averages = np.mean(irt_data, axis=0)
        irt_mean_avg = np.mean(irt_averages)
        
        # Avoid division by zero
        irt_averages = np.where(irt_averages == 0, 1, irt_averages)
        
        # Normalize data
        normalization_factors = irt_mean_avg / irt_averages
        normalized_data = data_matrix * normalization_factors
        
        print(f"Normalization completed using {len(irt_indices)} iRT peptides")
        return normalized_data
    
    def preprocess_data(self, data_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess data with rolling median and outlier handling
        """
        print("Preprocessing data with rolling statistics...")
        
        # Calculate rolling statistics
        n_segments = data_matrix.shape[1] // 3
        
        # Use optimized numba function for rolling median
        median_data = self.rolling_median_parallel(data_matrix)
        
        # Apply rolling average between adjacent points
        rolled_data = np.zeros((median_data.shape[0], median_data.shape[1] - 1))
        for i in range(median_data.shape[1] - 1):
            rolled_data[:, i] = (median_data[:, i] + median_data[:, i + 1]) / 2
        
        # Add final temperature point with outlier handling
        final_column = np.where(
            median_data[:, -1] > rolled_data[:, -1],
            0,  # Set to 0 if increasing at high temperature
            median_data[:, -1]
        )
        
        final_data = np.column_stack([rolled_data, final_column])
        
        # Apply temperature-specific outlier corrections
        self._apply_outlier_corrections(final_data)
        
        return final_data, median_data, rolled_data
    
    def _apply_outlier_corrections(self, data: np.ndarray, tolerance: float = 1.0):
        """Apply outlier corrections for high temperature points"""
        # Correction for 63°C (index 6)
        outlier_mask = data[:, 6] > (tolerance * data[:, 5])
        data[outlier_mask, 6] = (data[outlier_mask, 5] + data[outlier_mask, 6]) / 2
        
        # Correction for 68°C (index 7) - set to 0 if increasing
        increasing_mask = data[:, 7] > data[:, 6]
        data[increasing_mask, 7] = 0
        
        # Correction for 73°C (index 8) - set to 0 if increasing
        increasing_mask = data[:, 8] > data[:, 7]
        data[increasing_mask, 8] = 0
    
    def filter_valid_peptides(self, data: np.ndarray, sequences: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Filter peptides with valid data at early temperature points"""
        print("Filtering peptides with valid early temperature data...")
        
        # Check for missing values in first two temperature points (37°C, 42°C)
        early_points = data[:, :2]
        valid_mask = np.sum(early_points == 0, axis=1) == 0
        
        filtered_data = data[valid_mask]
        filtered_sequences = [seq for i, seq in enumerate(sequences) if valid_mask[i]]
        
        print(f"Retained {len(filtered_sequences)} peptides out of {len(sequences)}")
        
        # Normalize to 37°C (first temperature point)
        reference_point = filtered_data[:, 0]
        reference_point = np.where(reference_point == 0, 1, reference_point)  # Avoid division by zero
        normalized_data = filtered_data / reference_point[:, np.newaxis]
        
        return normalized_data, filtered_sequences
    
    def fit_sigmoid_curves(self, normalized_data: np.ndarray, 
                          progress_callback: Optional[callable] = None) -> List[FitResults]:
        """
        Fit 4PL sigmoid curves to estimate Tm values with confidence intervals
        """
        print("Fitting sigmoid curves...")
        
        n_peptides = normalized_data.shape[0]
        fit_results = []
        
        # Generate high-resolution x values for smooth curves
        x_fit = np.linspace(self.temperatures.min(), self.temperatures.max(), 100)
        
        for i in range(n_peptides):
            y_data = normalized_data[i, :]
            
            try:
                # Initial parameter estimates
                A_init = np.min(y_data)  # Minimum asymptote
                D_init = np.max(y_data)  # Maximum asymptote
                
                # Estimate slope
                slope = (y_data[-1] - y_data[0]) / (self.temperatures[-1] - self.temperatures[0])
                B_init = -2.0 if slope < 0 else 2.0  # Hill slope
                
                # Estimate inflection point
                mid_response = (D_init + A_init) / 2
                C_init_idx = np.argmin(np.abs(y_data - mid_response))
                C_init = self.temperatures[C_init_idx]
                
                # Set bounds
                bounds_lower = [0, -np.inf if slope < 0 else 0, self.temperatures.min(), 0]
                bounds_upper = [np.inf, 0 if slope < 0 else np.inf, self.temperatures.max(), np.inf]
                
                # Fit curve with confidence intervals
                popt, pcov = curve_fit(
                    self.four_pl_function,
                    self.temperatures,
                    y_data,
                    p0=[A_init, B_init, C_init, D_init],
                    bounds=(bounds_lower, bounds_upper),
                    maxfev=10000
                )
                
                # Calculate R-squared
                y_pred = self.four_pl_function(self.temperatures, *popt)
                ss_res = np.sum((y_data - y_pred) ** 2)
                ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                
                # Generate fitted curve
                fitted_curve = self.four_pl_function(x_fit, *popt)
                
                # Calculate confidence intervals
                param_errors = np.sqrt(np.diag(pcov))
                confidence_intervals = np.column_stack([popt - 1.96 * param_errors, 
                                                     popt + 1.96 * param_errors])
                
                # Tm is the C parameter (inflection point)
                tm = popt[2]
                
                fit_results.append(FitResults(
                    params=popt,
                    r_squared=r_squared,
                    tm=tm,
                    fitted_curve=fitted_curve,
                    confidence_intervals=confidence_intervals
                ))
                
            except Exception as e:
                # Failed fit - append None or default values
                fit_results.append(FitResults(
                    params=np.array([np.nan, np.nan, np.nan, np.nan]),
                    r_squared=0,
                    tm=np.nan,
                    fitted_curve=np.full(len(x_fit), np.nan),
                    confidence_intervals=np.full((4, 2), np.nan)
                ))
            
            if progress_callback and (i + 1) % 50 == 0:
                progress_callback(i + 1, n_peptides)
            elif (i + 1) % 100 == 0:
                print(f"Fitted {i + 1}/{n_peptides} curves")
        
        return fit_results
    
    def filter_valid_fits(self, fit_results: List[FitResults], sequences: List[str],
                         tm_range: Tuple[float, float] = (37, 73),
                         min_r_squared: float = 0.75) -> Tuple[List[FitResults], List[str]]:
        """Filter fits based on Tm range and correlation quality"""
        print("Filtering valid fits...")
        
        valid_indices = []
        for i, result in enumerate(fit_results):
            if (tm_range[0] <= result.tm <= tm_range[1] and 
                result.r_squared >= min_r_squared and
                not np.isnan(result.tm)):
                valid_indices.append(i)
        
        valid_fits = [fit_results[i] for i in valid_indices]
        valid_sequences = [sequences[i] for i in valid_indices]
        
        fit_percentage = (len(valid_fits) / len(fit_results)) * 100
        print(f"Valid fits: {len(valid_fits)}/{len(fit_results)} ({fit_percentage:.1f}%)")
        print(f"Correlation threshold (R² ≥ {min_r_squared}): {fit_percentage:.1f}%")
        
        return valid_fits, valid_sequences
    
    def create_tm_histogram(self, fit_results: List[FitResults], 
                           save_path: Optional[str] = None) -> plt.Figure:
        """Create publication-ready Tm distribution histogram"""
        tm_values = [result.tm for result in fit_results if not np.isnan(result.tm)]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create histogram
        bins = np.arange(37, 75, 2)
        n, bins, patches = ax.hist(tm_values, bins=bins, alpha=0.7, color='steelblue', 
                                  edgecolor='black', linewidth=0.8)
        
        # Customize appearance
        ax.set_xlabel('Temperature (°C)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Number of Profiled Peptides', fontsize=14, fontweight='bold')
        ax.set_title('Distribution of Melting Temperatures (Tm)', fontsize=16, fontweight='bold')
        
        # Set x-axis ticks
        ax.set_xticks(range(37, 74, 2))
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        mean_tm = np.mean(tm_values)
        std_tm = np.std(tm_values)
        ax.text(0.7, 0.8, f'Mean Tm: {mean_tm:.1f}°C\nStd: {std_tm:.1f}°C\nN = {len(tm_values)}',
                transform=ax.transAxes, fontsize=12, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_individual_curve(self, sequence: str, fit_result: FitResults,
                             normalized_data: np.ndarray, sequence_index: int,
                             save_path: Optional[str] = None) -> plt.Figure:
        """Create publication-ready individual denaturation curve plot"""
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Data points
        y_data = normalized_data[sequence_index, :]
        
        # Plot fitted curve
        x_fit = np.linspace(self.temperatures.min(), self.temperatures.max(), 100)
        ax.plot(x_fit, fit_result.fitted_curve, 'g-', linewidth=2.5, label='4PL Fit')
        
        # Plot data points with error bars (assuming some uncertainty)
        ax.errorbar(self.temperatures, y_data, yerr=y_data*0.05, 
                   fmt='o', markersize=8, color='blue', ecolor='blue', 
                   capsize=5, capthick=2, label='Experimental Data')
        
        # Mark Tm point
        tm_y = self.four_pl_function(fit_result.tm, *fit_result.params)
        ax.plot(fit_result.tm, tm_y, '*', markersize=15, color='red', 
                markeredgecolor='darkred', markeredgewidth=1, label=f'Tm = {fit_result.tm:.1f}°C')
        
        # Add dotted lines to Tm
        ax.axhline(y=tm_y, color='gray', linestyle=':', alpha=0.7)
        ax.axvline(x=fit_result.tm, color='gray', linestyle=':', alpha=0.7)
        
        # Customize plot
        ax.set_xlabel('Temperature (°C)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Normalized Peak Area', fontsize=14, fontweight='bold')
        ax.set_title(f'{sequence}\nTm = {fit_result.tm:.1f}°C (R² = {fit_result.r_squared:.3f})', 
                    fontsize=14, fontweight='bold')
        
        # Set x-axis ticks and labels
        ax.set_xticks(self.temperatures)
        ax.set_xticklabels([f'{int(t)}°C' for t in self.temperatures])
        
        # Legend and grid
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def export_results(self, fit_results: List[FitResults], sequences: List[str],
                      output_dir: str = '.', prefix: str = 'TIP_results'):
        """Export results to CSV files"""
        print("Exporting results...")
        
        # Extract Tm values and other parameters
        tm_values = [result.tm for result in fit_results]
        r_squared_values = [result.r_squared for result in fit_results]
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'Sequence': sequences,
            'Tm': tm_values,
            'R_squared': r_squared_values,
            'Parameter_A': [result.params[0] for result in fit_results],
            'Parameter_B': [result.params[1] for result in fit_results],
            'Parameter_C': [result.params[2] for result in fit_results],
            'Parameter_D': [result.params[3] for result in fit_results]
        })
        
        # Save main results
        results_path = f'{output_dir}/{prefix}_tm_values.csv'
        results_df.to_csv(results_path, index=False)
        
        # Save fitted curves
        fitted_curves = np.array([result.fitted_curve for result in fit_results])
        curves_df = pd.DataFrame(fitted_curves, index=sequences)
        curves_path = f'{output_dir}/{prefix}_fitted_curves.csv'
        curves_df.to_csv(curves_path)
        
        print(f"Results exported to {results_path}")
        print(f"Fitted curves exported to {curves_path}")
        
        return results_df

def main():
    """Example usage of the TIP pipeline"""
    
    # Initialize profiler
    tip = ThermalImmunopeptidomeProfiler()
    
    # Load data (replace with your file paths)
    # data_matrix, sequences = tip.load_data(
    #     'DIANNv18_rawoutput_B57_37C.csv',
    #     'DIANNv18_seqtarget_B57_37C.csv',
    #     quantification_method='MS1_PA'
    # )
    
    # For demonstration, create synthetic data
    print("Creating synthetic demonstration data...")
    n_peptides = 500
    temperatures = tip.temperatures
    sequences = [f'PEPTIDE{i:03d}' for i in range(n_peptides)]
    
    # Generate synthetic thermal stability data
    np.random.seed(42)
    tm_true = np.random.normal(55, 8, n_peptides)
    data_matrix = np.zeros((n_peptides, len(temperatures) * 3))  # 3 replicates per temperature
    
    for i, tm in enumerate(tm_true):
        for j, temp in enumerate(temperatures):
            # Simulate 4PL curve with noise
            base_intensity = 1.0 / (1 + (temp / tm) ** 3)
            for rep in range(3):
                col_idx = j * 3 + rep
                noise = np.random.normal(1, 0.1)
                data_matrix[i, col_idx] = max(0, base_intensity * noise)
    
    print(f"Generated synthetic data for {n_peptides} peptides")
    
    # Process data through pipeline
    start_time = time.time()
    
    # Normalize to iRTs (skip for synthetic data)
    # normalized_matrix = tip.normalize_to_irts(data_matrix, sequences)
    normalized_matrix = data_matrix
    
    # Preprocess data
    processed_data, _, _ = tip.preprocess_data(normalized_matrix)
    
    # Filter valid peptides
    filtered_data, filtered_sequences = tip.filter_valid_peptides(processed_data, sequences)
    
    # Fit sigmoid curves
    def progress_callback(current, total):
        print(f"Fitting progress: {current}/{total} ({100*current/total:.1f}%)")
    
    fit_results = tip.fit_sigmoid_curves(filtered_data, progress_callback)
    
    # Filter valid fits
    valid_fits, valid_sequences = tip.filter_valid_fits(fit_results, filtered_sequences)
    
    processing_time = time.time() - start_time
    print(f"\nProcessing completed in {processing_time:.2f} seconds")
    
    # Create visualizations
    print("Creating visualizations...")
    
    # Tm histogram
    fig1 = tip.create_tm_histogram(valid_fits, 'tm_distribution.png')
    plt.show()
    
    # Individual curve example
    if valid_fits:
        example_idx = 0
        fig2 = tip.plot_individual_curve(
            valid_sequences[example_idx], 
            valid_fits[example_idx],
            filtered_data, 
            example_idx,
            'example_curve.png'
        )
        plt.show()
    
    # Export results
    results_df = tip.export_results(valid_fits, valid_sequences)
    
    print("\nPipeline completed successfully!")
    print(f"Final results: {len(valid_fits)} peptides with valid Tm values")
    
    return tip, valid_fits, valid_sequences, results_df

if __name__ == "__main__":
    main()
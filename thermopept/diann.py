"""
Thermal Immunopeptidome Profiler (TIP) - Complete Demo Script
Demonstrates the full pipeline with performance benchmarks and publication-ready visualizations

Run this script to see the TIP pipeline in action with synthetic data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Set publication style
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 18,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3
})

def generate_realistic_thermal_data(n_peptides: int = 1000, 
                                   noise_level: float = 0.15,
                                   missing_data_rate: float = 0.05) -> tuple:
    """
    Generate realistic thermal stability data with biological variability
    """
    print(f"Generating realistic thermal data for {n_peptides} peptides...")
    
    temperatures = np.array([37, 42, 46, 50, 54, 58, 63, 68, 73])
    n_temps = len(temperatures)
    
    # Generate peptide sequences
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    sequences = []
    for i in range(n_peptides):
        # Generate 8-11 mer peptides (typical HLA length)
        length = np.random.choice([8, 9, 10, 11], p=[0.1, 0.6, 0.25, 0.05])
        seq = ''.join(np.random.choice(list(amino_acids), length))
        sequences.append(seq)
    
    # Generate biologically realistic Tm values
    # Most peptides stable around 50-60°C with some outliers
    tm_true = np.random.multivariate_normal(
        [55], [[64]], n_peptides
    ).flatten()
    
    # Clip to reasonable range
    tm_true = np.clip(tm_true, 40, 70)
    
    # Generate data matrix (3 replicates per temperature)
    data_matrix = np.zeros((n_peptides, n_temps * 3))
    
    for i, tm in enumerate(tm_true):
        # Biological parameters with variability
        A = np.random.normal(0.05, 0.02)  # Minimum with noise
        A = max(0.01, A)
        
        D = np.random.normal(1.0, 0.1)   # Maximum with noise
        D = max(0.5, D)
        
        B = np.random.normal(-2.5, 0.5)  # Hill slope with variability
        
        for j, temp in enumerate(temperatures):
            # 4PL model with biological noise
            base_intensity = D + (A - D) / (1 + (temp / tm) ** B)
            
            for rep in range(3):
                col_idx = j * 3 + rep
                
                # Add various noise sources:
                # 1. Measurement noise (15% CV)
                measurement_noise = np.random.normal(1, noise_level)
                
                # 2. Replicate variability (5% CV)
                replicate_noise = np.random.normal(1, 0.05)
                
                # 3. Systematic errors at high temperatures
                if temp > 60:
                    systematic_error = np.random.normal(1, 0.02)
                else:
                    systematic_error = 1
                
                final_intensity = base_intensity * measurement_noise * replicate_noise * systematic_error
                
                # Ensure non-negative values
                data_matrix[i, col_idx] = max(0, final_intensity)
                
                # Introduce missing data
                if np.random.random() < missing_data_rate:
                    data_matrix[i, col_idx] = 0
    
    print(f"Generated data with Tm range: {tm_true.min():.1f} - {tm_true.max():.1f}°C")
    return data_matrix, sequences, tm_true

def benchmark_performance():
    """
    Benchmark the performance improvements of the Python implementation
    """
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARK")
    print("="*60)
    
    # Test with different dataset sizes
    sizes = [100, 500, 1000, 2000]
    times = []
    
    for size in sizes:
        print(f"\nTesting with {size} peptides...")
        
        # Generate test data
        data_matrix, sequences, _ = generate_realistic_thermal_data(size, noise_level=0.1)
        
        # Initialize profiler
        from tip_pipeline import ThermalImmunopeptidomeProfiler  # Would import from main module
        tip = ThermalImmunopeptidomeProfiler()
        
        # Benchmark the pipeline
        start_time = time.time()
        
        # Preprocess
        processed_data, _, _ = tip.preprocess_data(data_matrix)
        
        # Filter
        filtered_data, filtered_sequences = tip.filter_valid_peptides(processed_data, sequences)
        
        # Fit curves (most computationally intensive step)
        fit_results = tip.fit_sigmoid_curves(filtered_data)
        
        # Filter results
        valid_fits, valid_sequences = tip.filter_valid_fits(fit_results, filtered_sequences)
        
        elapsed_time = time.time() - start_time
        times.append(elapsed_time)
        
        print(f"  ✓ Processed {len(valid_fits)} valid peptides in {elapsed_time:.2f}s")
        print(f"  ✓ Rate: {len(valid_fits)/elapsed_time:.1f} peptides/second")
    
    # Plot performance scaling
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(sizes, times, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Number of Peptides')
    plt.ylabel('Processing Time (seconds)')
    plt.title('Processing Time vs Dataset Size')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    rates = [s/t for s, t in zip(sizes, times)]
    plt.plot(sizes, rates, 's-', linewidth=2, markersize=8, color='orange')
    plt.xlabel('Number of Peptides')
    plt.ylabel('Processing Rate (peptides/sec)')
    plt.title('Processing Rate vs Dataset Size')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('performance_benchmark.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return dict(zip(sizes, times))

def create_publication_plots(tip, valid_fits, valid_sequences, filtered_data, tm_true=None):
    """
    Create comprehensive publication-ready visualizations
    """
    print("\n" + "="*60)
    print("CREATING PUBLICATION-READY VISUALIZATIONS")
    print("="*60)
    
    # Extract data
    tm_values = [result.tm for result in valid_fits]
    r_squared_values = [result.r_squared for result in valid_fits]
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Tm distribution histogram
    ax1 = plt.subplot(2, 3, 1)
    bins = np.arange(35, 75, 2)
    n, bins, patches = ax1.hist(tm_values, bins=bins, alpha=0.7, color='steelblue', 
                               edgecolor='black', linewidth=0.8)
    
    # Color code bars by frequency
    cm = plt.cm.viridis
    norm = plt.Normalize(vmin=n.min(), vmax=n.max())
    for i, p in enumerate(patches):
        p.set_facecolor(cm(norm(n[i])))
    
    ax1.set_xlabel('Melting Temperature (°C)')
    ax1.set_ylabel('Number of Peptides')
    ax1.set_title('Distribution of Melting Temperatures')
    
    # Add statistics
    mean_tm = np.mean(tm_values)
    std_tm = np.std(tm_values)
    ax1.axvline(mean_tm, color='red', linestyle='--', linewidth=2, alpha=0.8)
    ax1.text(0.7, 0.8, f'μ = {mean_tm:.1f}°C\nσ = {std_tm:.1f}°C\nN = {len(tm_values)}',
             transform=ax1.transAxes, fontsize=11, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. R² distribution
    ax2 = plt.subplot(2, 3, 2)
    ax2.hist(r_squared_values, bins=30, alpha=0.7, color='lightcoral', 
             edgecolor='black', linewidth=0.8)
    ax2.set_xlabel('R² Value')
    ax2.set_ylabel('Number of Peptides')
    ax2.set_title('Distribution of Fit Quality (R²)')
    ax2.axvline(np.mean(r_squared_values), color='darkred', linestyle='--', linewidth=2)
    
    # 3. Tm vs R² scatter plot
    ax3 = plt.subplot(2, 3, 3)
    scatter = ax3.scatter(tm_values, r_squared_values, alpha=0.6, c=r_squared_values, 
                         cmap='RdYlBu', s=30)
    ax3.set_xlabel('Melting Temperature (°C)')
    ax3.set_ylabel('R² Value')
    ax3.set_title('Fit Quality vs Melting Temperature')
    plt.colorbar(scatter, ax=ax3, label='R²')
    
    # Add correlation line
    correlation = np.corrcoef(tm_values, r_squared_values)[0, 1]
    ax3.text(0.05, 0.95, f'r = {correlation:.3f}', transform=ax3.transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 4. Example denaturation curves (best fits)
    ax4 = plt.subplot(2, 3, 4)
    
    # Select top 5 curves by R²
    top_indices = np.argsort(r_squared_values)[-5:]
    colors = plt.cm.Set2(np.linspace(0, 1, 5))
    
    x_fit = np.linspace(37, 73, 100)
    
    for i, idx in enumerate(top_indices):
        fit_result = valid_fits[idx]
        sequence = valid_sequences[idx]
        
        # Plot fitted curve
        ax4.plot(x_fit, fit_result.fitted_curve, color=colors[i], 
                linewidth=2, label=f'{sequence[:8]}... (R²={fit_result.r_squared:.3f})')
        
        # Plot data points
        y_data = filtered_data[idx, :]
        ax4.scatter(tip.temperatures, y_data, color=colors[i], s=30, alpha=0.7)
    
    ax4.set_xlabel('Temperature (°C)')
    ax4.set_ylabel('Normalized Intensity')
    ax4.set_title('Example Denaturation Curves (Top 5 Fits)')
    ax4.legend(fontsize=9, loc='upper right')
    
    # 5. Thermal stability landscape
    ax5 = plt.subplot(2, 3, 5)
    
    # Create 2D histogram of Tm vs sequence length
    seq_lengths = [len(seq) for seq in valid_sequences]
    
    hist, xedges, yedges = np.histogram2d(seq_lengths, tm_values, bins=[4, 20])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    
    im = ax5.imshow(hist.T, extent=extent, aspect='auto', origin='lower', 
                   cmap='plasma', interpolation='bilinear')
    ax5.set_xlabel('Sequence Length (aa)')
    ax5.set_ylabel('Melting Temperature (°C)')
    ax5.set_title('Thermal Stability Landscape')
    plt.colorbar(im, ax=ax5, label='Peptide Count')
    
    # 6. Comparison with true values (if available)
    ax6 = plt.subplot(2, 3, 6)
    
    if tm_true is not None:
        # Match sequences to get corresponding true values
        true_tm_matched = []
        for seq in valid_sequences:
            seq_idx = [i for i, s in enumerate(sequences) if s == seq]
            if seq_idx:
                true_tm_matched.append(tm_true[seq_idx[0]])
        
        if true_tm_matched:
            ax6.scatter(true_tm_matched, tm_values, alpha=0.6, s=30)
            
            # Perfect correlation line
            min_tm, max_tm = min(min(true_tm_matched), min(tm_values)), max(max(true_tm_matched), max(tm_values))
            ax6.plot([min_tm, max_tm], [min_tm, max_tm], 'r--', linewidth=2, alpha=0.8)
            
            # Calculate correlation
            correlation = np.corrcoef(true_tm_matched, tm_values)[0, 1]
            rmse = np.sqrt(np.mean((np.array(true_tm_matched) - np.array(tm_values))**2))
            
            ax6.set_xlabel('True Tm (°C)')
            ax6.set_ylabel('Fitted Tm (°C)')
            ax6.set_title('True vs Fitted Melting Temperatures')
            ax6.text(0.05, 0.95, f'r = {correlation:.3f}\nRMSE = {rmse:.1f}°C', 
                    transform=ax6.transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        # Alternative plot: Tm vs fit parameters
        param_B = [result.params[1] for result in valid_fits]
        ax6.scatter(tm_values, param_B, alpha=0.6, s=30, c=r_squared_values, cmap='viridis')
        ax6.set_xlabel('Melting Temperature (°C)')
        ax6.set_ylabel('Hill Slope (Parameter B)')
        ax6.set_title('Tm vs Hill Slope Relationship')
    
    plt.tight_layout()
    plt.savefig('comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def main():
    """
    Complete demonstration of the TIP pipeline
    """
    print("🧬 THERMAL IMMUNOPEPTIDOME PROFILER (TIP) - PYTHON DEMO")
    print("="*70)
    
    # Step 1: Generate realistic data
    print("\n📊 STEP 1: DATA GENERATION")
    data_matrix, sequences, tm_true = generate_realistic_thermal_data(
        n_peptides=800, 
        noise_level=0.12,
        missing_data_rate=0.03
    )
    
    # Step 2: Initialize and run pipeline
    print("\n⚙️  STEP 2: PIPELINE EXECUTION")
    
    # Import the main TIP class (in real usage, this would be from the main module)
    # For demo purposes, we'll create a simplified version
    from tip_pipeline import ThermalImmunopeptidomeProfiler
    
    tip = ThermalImmunopeptidomeProfiler()
    
    start_time = time.time()
    
    # Process data
    print("  - Preprocessing data...")
    processed_data, _, _ = tip.preprocess_data(data_matrix)
    
    print("  - Filtering valid peptides...")
    filtered_data, filtered_sequences = tip.filter_valid_peptides(processed_data, sequences)
    
    print("  - Fitting sigmoid curves...")
    fit_results = tip.fit_sigmoid_curves(filtered_data)
    
    print("  - Filtering valid fits...")
    valid_fits, valid_sequences = tip.filter_valid_fits(fit_results, filtered_sequences)
    
    processing_time = time.time() - start_time
    
    # Step 3: Results summary
    print("\n📈 STEP 3: RESULTS SUMMARY")
    print(f"  ✓ Total processing time: {processing_time:.2f} seconds")
    print(f"  ✓ Input peptides: {len(sequences)}")
    print(f"  ✓ Valid fits: {len(valid_fits)}")
    print(f"  ✓ Success rate: {100*len(valid_fits)/len(sequences):.1f}%")
    print(f"  ✓ Processing rate: {len(valid_fits)/processing_time:.1f} peptides/second")
    
    if valid_fits:
        tm_values = [result.tm for result in valid_fits]
        r_squared_values = [result.r_squared for result in valid_fits]
        
        print(f"  ✓ Tm range: {min(tm_values):.1f} - {max(tm_values):.1f}°C")
        print(f"  ✓ Mean Tm: {np.mean(tm_values):.1f} ± {np.std(tm_values):.1f}°C")
        print(f"  ✓ Mean R²: {np.mean(r_squared_values):.3f}")
        print(f"  ✓ High quality fits (R² > 0.9): {sum(r > 0.9 for r in r_squared_values)}/{len(r_squared_values)}")
    
    # Step 4: Visualizations
    print("\n🎨 STEP 4: CREATING VISUALIZATIONS")
    
    # Basic plots
    fig1 = tip.create_tm_histogram(valid_fits)
    plt.savefig('tm_histogram_demo.png', dpi=300, bbox_inches='tight')
    
    # Individual curve example
    if valid_fits:
        best_fit_idx = np.argmax([r.r_squared for r in valid_fits])
        fig2 = tip.plot_individual_curve(
            valid_sequences[best_fit_idx], 
            valid_fits[best_fit_idx],
            filtered_data, 
            best_fit_idx
        )
        plt.savefig('best_curve_demo.png', dpi=300, bbox_inches='tight')
    
    # Comprehensive analysis
    fig3 = create_publication_plots(tip, valid_fits, valid_sequences, filtered_data, tm_true)
    
    # Step 5: Export results
    print("\n💾 STEP 5: EXPORTING RESULTS")
    results_df = tip.export_results(valid_fits, valid_sequences, prefix='demo')
    
    print(f"  ✓ Results exported to demo_tm_values.csv")
    print(f"  ✓ Fitted curves exported to demo_fitted_curves.csv")
    
    # Step 6: Performance benchmark
    print("\n⚡ STEP 6: PERFORMANCE BENCHMARK")
    benchmark_times = benchmark_performance()
    
    # Final summary
    print("\n🎯 PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("Generated files:")
    print("  - tm_histogram_demo.png")
    print("  - best_curve_demo.png") 
    print("  - comprehensive_analysis.png")
    print("  - performance_benchmark.png")
    print("  - demo_tm_values.csv")
    print("  - demo_fitted_curves.csv")
    
    return {
        'tip': tip,
        'valid_fits': valid_fits,
        'valid_sequences': valid_sequences,
        'results_df': results_df,
        'benchmark_times': benchmark_times,
        'processing_time': processing_time
    }

if __name__ == "__main__":
    # Run the complete demo
    results = main()
    
    # Additional analysis suggestions
    print("\n💡 ADDITIONAL ANALYSIS SUGGESTIONS:")
    print("  - Statistical comparison between peptide groups")
    print("  - Machine learning prediction of thermal stability")
    print("  - Sequence motif analysis for stability patterns") 
    print("  - Integration with structural prediction tools")
    print("  - Batch processing of multiple experiments")
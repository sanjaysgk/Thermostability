from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class DIADataProcessor(ABC):
    """
    Abstract base class for processing DIA-NN report.tsv files
    from multiple temperatures and experiments.
    """

    def __init__(self, temp_points=None, quant_type="MS1"):
        """
        Parameters
        ----------
        temp_points : list of int, optional
            Temperatures used in the experiment (e.g., [37,42,46,...])
        quant_type : str
            Quantification type, one of ['Precursor.Quantity', 'MS1', 'MS2', 'Total']
        """
        self.temp_points = temp_points or [37, 42, 46, 50, 54, 58, 63, 68, 73]
        self.quant_type = quant_type

    @abstractmethod
    def load_report(self, file_path):
        """
        Abstract method to load a DIA-NN report.tsv file.
        
        Returns
        -------
        pd.DataFrame
        """
        pass

    @abstractmethod
    def extract_peptides(self, data_frame):
        """
        Abstract method to extract peptide sequences and intensities
        according to quant_type.
        
        Returns
        -------
        peptide_list : list of str
        intensity_matrix : np.ndarray
        replicate_labels : list of str
        """
        pass

    @abstractmethod
    def normalize_iRT(self, intensity_matrix, peptide_list):
        """
        Abstract method to normalize the intensity data using iRT peptides.
        
        Returns
        -------
        np.ndarray
            Normalized intensity matrix
        """
        pass

    @abstractmethod
    def smooth_and_filter(self, intensity_matrix):
        """
        Abstract method to smooth the profiles, handle missing/outlier values.
        
        Returns
        -------
        np.ndarray
            Smoothed and filtered matrix
        """
        pass

    @abstractmethod
    def fit_sigmoid(self, intensity_matrix):
        """
        Abstract method to fit sigmoid curves to each peptide for Tm estimation.
        
        Returns
        -------
        Tm_values : np.ndarray
        fitted_curves : np.ndarray
        """
        pass

    @abstractmethod
    def filter_valid(self, Tm_values, intensity_matrix, peptide_list):
        """
        Abstract method to filter valid Tm values and peptides.
        
        Returns
        -------
        valid_Tm : np.ndarray
        filtered_matrix : np.ndarray
        filtered_peptides : list of str
        """
        pass

    @abstractmethod
    def save_results(self, Tm_values, peptides, fitted_curves, output_dir):
        """
        Abstract method to export results to CSV or other formats.
        """
        pass

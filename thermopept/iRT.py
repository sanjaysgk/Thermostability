import numpy as np

class iRTNormalizer:
    """
    A class to normalize peptide intensity data using iRT peptides.
    """

    def __init__(self, iRT_sequences=None):
        """
        Parameters
        ----------
        iRT_sequences : list of str
            List of known iRT peptide sequences for normalization.
            Defaults to commonly used iRTs.
        """
        if iRT_sequences is None:
            self.iRT_sequences = [
                'LGGNEQVTR', 'GAGSSEPVTGLDAK', 'VEATFGVDESNAK', 'YILAGVENSK',
                'TPVISGGPYEYR', 'TPVITGAPYEYR', 'DGLDAASYYAPVR', 'ADVTPADFSEWSK',
                'GTFIIDPGGVIR', 'GTFIIDPAAVIR', 'LFLQFGAQGSPFLK'
            ]
        else:
            self.iRT_sequences = iRT_sequences

    def normalize(self, data_matrix, peptide_list):
        """
        Normalize intensity data using iRT peptides.

        Parameters
        ----------
        data_matrix : np.ndarray
            Raw peptide intensity matrix (peptides x replicates/conditions)
        peptide_list : list of str
            List of peptide sequences corresponding to rows in data_matrix

        Returns
        -------
        np.ndarray
            Normalized data matrix
        """
        # Collect iRT rows
        iRT_rows = [peptide_list.index(seq) for seq in self.iRT_sequences if seq in peptide_list]
        if not iRT_rows:
            raise ValueError("No iRT sequences found in peptide list.")

        DiRT = data_matrix[iRT_rows, :]
        iRT_avr = np.mean(DiRT, axis=1, keepdims=True)
        DiRT_norm = DiRT / iRT_avr
        iRT_avr_mean = np.mean(DiRT_norm, axis=0)
        normalized_data = data_matrix / iRT_avr_mean

        return normalized_data

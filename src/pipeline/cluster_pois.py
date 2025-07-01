import logging
import numpy as np
import hdbscan
import pandas as pd
from typing import List

class ClustesSites:
    def __init__(
            self, 
            eps_km: float = 0.1,
            min_sample_size: int = 5,
            min_samples: int = 4):

        self.eps_radian = eps_km / 6371.0088
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_sample_size,
            min_samples=min_samples,
            metric="haversine",
            cluster_selection_epsilon=self.eps_radian)
        logging.info("(ClusterSites): Successfuly initialized HDBSCAN model.")
        

    def predict_clusters(self,coords_rad: np.ndarray) -> List:
        try:
            clusters = self.clusterer.fit_predict(coords_rad)
            logging.info(f"(ClusterSites): Successfully clustered sites.")
            return clusters
        except Exception as e:
            raise ValueError(f"Failed to create clusters.\n{e}")
        

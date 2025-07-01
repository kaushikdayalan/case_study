import os
import re
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict
def extract_cluster_bounding_boxes_dict(df: pd.DataFrame) -> Dict:
    cluster_boxes = {}
    buffer = 0.003

    for cluster_id in sorted(df["cluster"].unique()):
        if cluster_id == -1:
            continue

        cluster_points = df[df["cluster"] == cluster_id]
        min_lat = cluster_points["lat"].min() - buffer
        max_lat = cluster_points["lat"].max() + buffer
        min_lon = cluster_points["lon"].min() - buffer
        max_lon = cluster_points["lon"].max() + buffer

        cluster_boxes[cluster_id] = {
            "min_lat": min_lat,
            "min_lon": min_lon,
            "max_lat": max_lat,
            "max_lon": max_lon,
        }

    return cluster_boxes

def save_cluster_pois(all_pois: Dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    for cluster_id, pois in all_pois.items():
        file_path = os.path.join(output_dir, f"cluster_{cluster_id}.json")
        with open(file_path, "w") as file:
            json.dump(pois, file)

def save_site_pois(all_pois: Dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    for site_id, pois in all_pois.items():
        file_path = os.path.join(output_dir, f"site_{site_id}.json")
        with open(file_path, "w") as file:
            json.dump(pois, file)


def save_intermediete_data(
    df: pd.DataFrame, output_dir: str, file_name: str
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    if "csv" in file_name:
        df.to_csv(f"{output_dir}/{file_name}", index=False)
        logging.info(f"Succesfully saved file: {output_dir}/{file_name}")

    if "parquet" in file_name:
        df.to_parquet(f"{output_dir}/{file_name}", index=False)
        print(f"Succesfully saved file: {output_dir}/{file_name}")


def read_intermediate_data(file_path: str) -> pd.DataFrame:
    if ".csv" in file_path:
        return pd.read_csv(file_path)
    if ".parquet" in file_path:
        return pd.read_parquet(file_path)
    

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = (
        np.sin(dphi / 2) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    )
    return 2 * R * np.arcsin(np.sqrt(a))


def is_open_24_7(value):
    if not isinstance(value, str):
        return False
    val = value.strip().lower()
    if "24/7" in val:
        return True
    patterns = [
        r"\bmo-su\b.*00:00[-–]24:00",
        r"\bmo-su\b.*24:00",
        r"\b00:00[-–]24:00\b"
    ]
    return any(re.search(p, val) for p in patterns)
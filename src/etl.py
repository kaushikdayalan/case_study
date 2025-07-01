import ast
import logging
import numpy as np
from glob import glob
import pandas as pd
from tqdm import tqdm
from src.pipeline.poi_api import OverPassAPI
from src.pipeline.cluster_pois import ClustesSites
from src.pipeline.transforms import transformation_pipeline
from src.pipeline.enrich_sites import EnrichSites
from src.utils.helpers import (
    extract_cluster_bounding_boxes_dict,
    save_cluster_pois, save_site_pois
)
from typing import List, Dict


def clean_site_data(site_df: pd.DataFrame) -> pd.DataFrame:
    logging.info(f"Cleaning and removing outliers from site_data.")

    site_df["geoCoordinates"] = site_df["geoCoordinates"].apply(
        ast.literal_eval
    )
    site_df["lon"] = site_df["geoCoordinates"].apply(lambda x: x[0])
    site_df["lat"] = site_df["geoCoordinates"].apply(lambda x: x[1])

    GERMANY_BOUNDS = {
        "lat_min": 47.2,
        "lat_max": 55.1,
        "lon_min": 5.9,
        "lon_max": 15.0,
    }

    mask = (
        (site_df["lat"] >= GERMANY_BOUNDS["lat_min"])
        & (site_df["lat"] <= GERMANY_BOUNDS["lat_max"])
        & (site_df["lon"] >= GERMANY_BOUNDS["lon_min"])
        & (site_df["lon"] <= GERMANY_BOUNDS["lon_max"])
    )

    site_df["is_outlier"] = ~mask

    logging.info(
        f"The number of sites that lie outside the approximated border of Germany is: {len(site_df[site_df['is_outlier'] == True])}"
    )

    site_df = site_df[site_df["is_outlier"] == False]

    logging.info(f"Successfuly removed outliers if any.")
    return site_df


def extract_clusters(
    site_df: pd.DataFrame,
    eps_km: float,
    min_sample_size: int,
    min_samples: int,
) -> pd.DataFrame:

    
    clusterer = ClustesSites(
        eps_km=eps_km, min_sample_size=min_sample_size, min_samples=min_samples
    )
    coords_rad = np.radians(site_df[["lat", "lon"]].to_numpy())

    clusters = clusterer.predict_clusters(coords_rad=coords_rad)
    site_df["cluster"] = clusters
    return site_df


def extract_poi_data(site_df: pd.DataFrame) -> List[Dict]:
    cluster_pois = {}
    individual_sites = {}

    overpass_api = OverPassAPI()

    cluster_boxes = extract_cluster_bounding_boxes_dict(df=site_df)

    for cluster_id, bbox in tqdm(
        cluster_boxes.items(), desc="Querying cluster POIs"
    ):
        query = overpass_api.build_bbox_query(bbox)
        poi_response = overpass_api.query_overpass(query)
        cluster_pois[cluster_id] = poi_response

    unclustered_sites_df = site_df[site_df["cluster"] == -1]

    for lat, lon, site_id in tqdm(
        unclustered_sites_df[["lat", "lon", "id"]].values,
        desc="Querying individual sites",
    ):
        around_query = overpass_api.build_around_query(lat=lat, lon=lon)
        poi_response = overpass_api.query_overpass(query=around_query)
        individual_sites[site_id] = poi_response

    return cluster_pois, individual_sites


def run_etl_pipeline(
    site_df_path: str, eps_km: float, min_sample_size: int, min_samples: int, output_dir: str
):

    ### BRONZE LAYER

    site_df = pd.read_csv(site_df_path)

    site_df = clean_site_data(site_df=site_df)

    site_df = extract_clusters(
        site_df=site_df,
        eps_km=eps_km,
        min_sample_size=min_sample_size,
        min_samples=min_samples,
    )

    cluster_pois, individual_pois = extract_poi_data(site_df=site_df)

    save_cluster_pois(
        all_pois=cluster_pois, output_dir="./data_test/raw/cluster_pois/"
    )

    save_site_pois(
        all_pois=individual_pois, output_dir="./data_test/raw/site_pois/"
    )

    cluster_pois_file_paths = glob(
        "./data_test/raw/cluster_pois/*"
    )
    individual_pois_file_paths = glob(
        "./data_test/raw/site_pois/*"
    )
    all_pois = cluster_pois_file_paths + individual_pois_file_paths

    ### BRONZE LAYER

    ### SILVER LAYER

    return transformation_pipeline(
        site_df=site_df,
        poi_json_file_paths=all_pois,
        output_dir=output_dir,
    )
    ### SILVER LAYER

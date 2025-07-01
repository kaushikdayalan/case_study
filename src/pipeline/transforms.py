import os
import json
import logging
import pandas as pd
from glob import glob
from typing import List, Union
from src.utils.helpers import haversine, save_intermediete_data


def read_raw_data(poi_json_file_paths: List) -> List[pd.DataFrame]:
    fast_food_poi_df = []
    fuel_station_poi_df = []
    supermarket_poi_df = []

    for poi_json_file in poi_json_file_paths:
        with open(poi_json_file, "r") as file:
            poi_data = json.load(file)

        for poi in poi_data:
            tags = poi.get("tags", {})

            amenity = tags.get("amenity")
            shop = tags.get("shop")

            if amenity == "fast_food":
                df = pd.json_normalize(poi)
                df["category"] = "fast_food"
                fast_food_poi_df.append(df)

            elif amenity == "fuel":
                df = pd.json_normalize(poi)
                df["category"] = "fuel"
                fuel_station_poi_df.append(df)

            elif shop == "supermarket":
                df = pd.json_normalize(poi)
                df["category"] = "supermarket"
                supermarket_poi_df.append(df)

    fast_food_poi_df = pd.concat(fast_food_poi_df)
    fuel_station_poi_df = pd.concat(fuel_station_poi_df)
    supermarket_poi_df = pd.concat(supermarket_poi_df)

    logging.info(
        f"(Transforms): Succesfully read {len(poi_json_file_paths)} raw POI data file and converted to DataFrame."
    )
    return (fast_food_poi_df, fuel_station_poi_df, supermarket_poi_df)


def deduplicate_poi_data(df: pd.DataFrame) -> pd.DataFrame:
    if len(df.duplicated(subset=["type", "poi_id"])) > 0:
        logging.info(
            f"(Transforms): Found {len(df[df.duplicated(subset=['type', 'poi_id'])])} duplicates."
        )

        df = df.drop_duplicates(subset=["type", "poi_id"], keep="first")

        logging.info(f"(Transforms): Successfuly deduplicated the dataset.")
        return df
    else:
        logging.info(f"(Transforms): No duplicate POIs found.")
        return df


def extract_lat_long_for_ways(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts latitude and longitude for OSM 'way' elements by filling missing values
    from the corresponding 'center' coordinates.

    Args:
        df (pd.DataFrame): A DataFrame containing OSM elements with possible 'lat', 'lon',
                           'center.lat', and 'center.lon' columns.

    Returns:
        pd.DataFrame: The input DataFrame with new 'latitude' and 'longitude' columns,
                      where missing values from 'lat'/'lon' are filled using 'center.lat'/'center.lon'.
    """
    df["lat"] = df["lat"].fillna(df["center.lat"])
    df["lon"] = df["lon"].fillna(df["center.lon"])
    df = df.drop(["center.lat", "center.lon"], axis=1)
    logging.info(
        f"(Transforms): Successfully extracted latitudes and longitudes of ways(center)."
    )
    return df


def validate_lat(lat: float) -> Union[float, None]:
    """
    Returns the latitude if it's within the valid range (-90 to 90), else None.
    """
    try:
        lat = float(lat)
        if -90.0 <= lat <= 90.0:
            return lat
    except (ValueError, TypeError):
        pass
    logging.critical("Latitude is wrong.")
    return None


def validate_lon(lon: float) -> Union[float, None]:
    """
    Returns the longitude if it's within the valid range (-180 to 180), else None.
    """
    try:
        lon = float(lon)
        if -180.0 <= lon <= 180.0:
            return lon
    except (ValueError, TypeError):

        pass
    logging.critical("Longitude is wrong.")
    return None


def compute_distance_between_pois_sites(
    all_poi_data: pd.DataFrame, site_data: pd.DataFrame
) -> pd.DataFrame:
    all_poi_data["key"] = 1
    site_data["key"] = 1
    pairs = all_poi_data.merge(
        site_data, on="key", suffixes=("_site", "_poi")
    ).drop("key", axis=1)

    pairs["distance_m"] = haversine(
        pairs["lat_site"],
        pairs["lon_site"],
        pairs["lat_poi"],
        pairs["lon_poi"],
    )

    proximity_df = pairs.loc[
        pairs["distance_m"] <= 100,
        ["site_id", "poi_id", "distance_m", "category"],
    ]
    return proximity_df


def transformation_pipeline(
    site_df: pd.DataFrame, poi_json_file_paths: List, output_dir: str
) -> pd.DataFrame:

    ## Important columns in site data
    important_columns = [
        "id",
        "locality",
        "postalCode",
        "state",
        "operatorId",
        "operatorName",
        "lon",
        "lat",
    ]
    site_df = site_df[important_columns]
    site_df = site_df.rename({"id": "site_id"}, axis=1)

    ## Reading raw POI data and seperating them into category based DataFrames
    (fast_food_poi_df, fuel_station_poi_df, supermarket_poi_df) = (
        read_raw_data(poi_json_file_paths=poi_json_file_paths)
    )

    ## Renaming the column "id" to "poi_id"
    fast_food_poi_df = fast_food_poi_df.rename(columns={"id": "poi_id"})
    fuel_station_poi_df = fuel_station_poi_df.rename(columns={"id": "poi_id"})
    supermarket_poi_df = supermarket_poi_df.rename(columns={"id": "poi_id"})

    ## Deduplicating POI data
    fast_food_poi_df = deduplicate_poi_data(df=fast_food_poi_df)
    fuel_station_poi_df = deduplicate_poi_data(df=fuel_station_poi_df)
    supermarket_poi_df = deduplicate_poi_data(df=supermarket_poi_df)

    ## Extracting Latitude and Longitude for ways type data
    fast_food_poi_df = extract_lat_long_for_ways(df=fast_food_poi_df)
    fuel_station_poi_df = extract_lat_long_for_ways(df=fuel_station_poi_df)
    supermarket_poi_df = extract_lat_long_for_ways(df=supermarket_poi_df)

    ## Validating Latitude and Longitude values
    fast_food_poi_df["lat"] = fast_food_poi_df["lat"].apply(validate_lat)
    fast_food_poi_df["lon"] = fast_food_poi_df["lon"].apply(validate_lon)

    fuel_station_poi_df["lat"] = fuel_station_poi_df["lat"].apply(validate_lat)
    fuel_station_poi_df["lon"] = fuel_station_poi_df["lon"].apply(validate_lon)

    supermarket_poi_df["lat"] = supermarket_poi_df["lat"].apply(validate_lat)
    supermarket_poi_df["lon"] = supermarket_poi_df["lon"].apply(validate_lon)

    ## Saving fast food POI data
    fast_food_poi_df_file_name = (
        f"{fast_food_poi_df.category.values[0]}_pois.parquet"
    )
    save_intermediete_data(
        df=fast_food_poi_df,
        output_dir=output_dir,
        file_name=fast_food_poi_df_file_name,
    )

    ## Saving fuel stations POI data
    fuel_station_poi_df_file_name = (
        f"{fuel_station_poi_df.category.values[0]}_pois.parquet"
    )
    save_intermediete_data(
        df=fuel_station_poi_df,
        output_dir=output_dir,
        file_name=fuel_station_poi_df_file_name,
    )

    ## Saving supermarket POI data
    supermarket_poi_df_file_name = (
        f"{supermarket_poi_df.category.values[0]}_pois.parquet"
    )
    save_intermediete_data(
        df=supermarket_poi_df,
        output_dir=output_dir,
        file_name=supermarket_poi_df_file_name,
    )

    ## Saving cleaned site data
    site_data_file_name = "site_data.parquet"
    save_intermediete_data(
        df=site_df, output_dir=output_dir, file_name=site_data_file_name
    )

    
    staging_pois_files = glob(f"{output_dir}/*_pois.parquet")
    common_columns = ["poi_id", "lat", "lon", "category"]

    all_pois_data = [
        pd.read_parquet(file)[common_columns]
        for file in staging_pois_files
    ]
    all_pois_data = pd.concat(all_pois_data)


    ## Creating proximity relationship dataset between site and POI
    proximity_df = compute_distance_between_pois_sites(
        all_poi_data=all_pois_data, site_data=site_df
    )

    ## Saving proximity data
    proximity_data_file_name = "site_pois_proximities.parquet"
    save_intermediete_data(
        df=proximity_df,
        output_dir=output_dir,
        file_name=proximity_data_file_name,
    )

    return (
        site_data_file_name,
        fast_food_poi_df_file_name,
        fuel_station_poi_df_file_name,
        supermarket_poi_df_file_name,
        proximity_data_file_name,
    )

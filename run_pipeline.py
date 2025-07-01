import os
import logging
logging.getLogger().setLevel(logging.INFO)

import warnings
warnings.filterwarnings("ignore")


from src.etl import run_etl_pipeline
from src.pipeline.enrich_sites import EnrichSites
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Input to run pipeline.")

    parser.add_argument(
        "--site_df_path",
        type=str,
        required=True,
        help="The directory where output should be stored.",
    )

    parser.add_argument(
    "--eps_km",
    type=float,
    default=0.1,
    help="(HDBSCAN model) The epsilon value to apply to the algorithm.",
)

    parser.add_argument(
        "--min_sample_size",
        type=int,
        default=5,
        help="(HDBSCAN model) The minimum number of samples a cluster should have to be considered a cluster.",
    )

    parser.add_argument(
        "--min_samples",
        type=int,
        default=4,
        help="(HDBSCAN model) The minimum number of samples the center of a cluster should have to be considered as a center of the cluster.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="path where intermediate/staging data should be stored.",
    )

    parser.add_argument(
        "--enriched_dataset_save_path",
        type=str,
        required=True,
        help="path where the enriched dataset should be saved",
    )

    args = parser.parse_args()

    site_df_path = args.site_df_path
    output_dir = args.output_dir
    enriched_dataset_save_path = args.enriched_dataset_save_path
    eps_km = args.eps_km
    min_sample_size = args.min_sample_size
    min_samples = args.min_samples


    if args.eps_km == 0.1:
        logging.info("No '--eps_km' provided. Using default: 0.0")

    if args.min_sample_size == 7:
        logging.info("No '--min_sample_size' provided. Using default: 5")
    
        
    if args.min_samples == 4:
        logging.info("No '--min_samples' provided. Using default: same as 4")
       
    (
        site_data_file_name,
        fast_food_poi_df_file_name,
        fuel_station_poi_df_file_name,
        supermarket_poi_df_file_name,
        proximity_data_file_name,
    ) = run_etl_pipeline(
        site_df_path=site_df_path,
        eps_km=eps_km,
        min_sample_size=min_sample_size,
        min_samples=min_samples,
        output_dir=output_dir,
    )

    ## GOLDEN LAYER ##

    enrich_sites = EnrichSites(
        site_data_path=f"{output_dir}/{site_data_file_name}",
        fast_food_data_path=f"{output_dir}/{fast_food_poi_df_file_name}",
        fuel_data_path=f"{output_dir}/{fuel_station_poi_df_file_name}",
        supermarket_data_path=f"{output_dir}/{supermarket_poi_df_file_name}",
        proximity_data_path=f"{output_dir}/{proximity_data_file_name}",
    )

    enirched_site_data = enrich_sites.enirch_site_data_with_features()

    print(enirched_site_data.head())

    os.makedirs(enriched_dataset_save_path, exist_ok=True)
    enirched_site_data.to_csv(
        f"{enriched_dataset_save_path}/enriched_site_data.csv", index=False
    )

    logging.info(f"Enriched dataset saved here: {enriched_dataset_save_path}/enriched_site_data.csv")

    ## GOLDEN LAYER ##
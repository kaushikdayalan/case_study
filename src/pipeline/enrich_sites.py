import numpy as np
import pandas as pd
from src.utils.helpers import read_intermediate_data, is_open_24_7


class EnrichSites:
    def __init__(
        self,
        site_data_path: str,
        fast_food_data_path: str,
        supermarket_data_path: str,
        fuel_data_path: str,
        proximity_data_path: str,
    ):

        self.site_data = read_intermediate_data(file_path=site_data_path)
        self.fast_food_data = read_intermediate_data(file_path=fast_food_data_path)
        self.supermarket_data = read_intermediate_data(
            file_path=supermarket_data_path
        )
        self.fuel_data = read_intermediate_data(file_path=fuel_data_path)
        self.proximity_data = read_intermediate_data(
            file_path=proximity_data_path
        )
        self.site_enriched_data = self.site_data

    def site_poi_counts(self) -> pd.DataFrame:
        return (
            self.proximity_data.groupby("site_id")
            .size()
            .reset_index(name="total_num_pois")
        )

    def poi_category_counts(self) -> pd.DataFrame:
        return (
            self.proximity_data.groupby(["site_id", "category"])
            .size()
            .unstack(fill_value=0)
            .rename(
                columns={
                    "supermarket": "num_supermarkets",
                    "fast_food": "num_fast_food",
                    "fuel": "num_fuel_stations",
                }
            )
        )

    def closest_poi_category(self) -> pd.DataFrame:
        closest_rows = self.proximity_data.loc[
            self.proximity_data.groupby("site_id")["distance_m"].idxmin()
        ]

        closest_category = closest_rows[["site_id", "category"]].rename(
            columns={"category": "closest_category"}
        )
        return closest_category

    def extract_poi_data_features(self):
        poi_dfs = [self.fast_food_data, self.fuel_data, self.supermarket_data]
        categories = ["fast_food", "fuel", "supermarket"]

        poi_feature_dfs = []

        for df, category in zip(poi_dfs, categories):
            df = df.copy()
            df["poi_id"] = df["poi_id"]
            df["category"] = category

            if "tags.toilets" in df.columns:
                df["has_toilet"] = (
                    df["tags.toilets"].fillna("").str.lower() == "yes"
                )
            else:
                df["has_toilet"] = False

            internet = (
                df["tags.internet_access"].fillna("")
                if "tags.internet_access" in df.columns
                else pd.Series("", index=df.index)
            )
            wifi = (
                df["tags.wifi"].fillna("")
                if "tags.wifi" in df.columns
                else pd.Series("", index=df.index)
            )
            df["has_wifi"] = internet.str.contains(
                "yes|wlan", case=False, na=False
            ) | wifi.str.lower().eq("yes")

            indoor = (
                df["tags.indoor_seating"].fillna("")
                if "tags.indoor_seating" in df.columns
                else pd.Series("", index=df.index)
            )
            outdoor = (
                df["tags.outdoor_seating"].fillna("")
                if "tags.outdoor_seating" in df.columns
                else pd.Series("", index=df.index)
            )
            df["has_seating"] = indoor.str.lower().eq(
                "yes"
            ) | outdoor.str.lower().eq("yes")

            if "tags.opening_hours" in df.columns:
                df["open_24hr"] = df["tags.opening_hours"].apply(is_open_24_7)
            else:
                df["open_24hr"] = False

            df["tags.brand"] = (
                df["tags.brand"] if "tags.brand" in df.columns else np.nan
            )

            poi_feature_dfs.append(
                df[
                    [
                        "poi_id",
                        "category",
                        "has_toilet",
                        "has_wifi",
                        "has_seating",
                        "open_24hr",
                        "tags.brand",
                    ]
                ]
            )

        poi_features_df = pd.concat(poi_feature_dfs, ignore_index=True)
        return poi_features_df

    def enirch_site_data_with_features(self) -> pd.DataFrame:
        ## Enrich with total POI counts
        self.site_enriched_data = self.site_enriched_data.merge(
            self.site_poi_counts(), on="site_id", how="left"
        )
        self.site_enriched_data = self.site_enriched_data.fillna(0)

        ## Enrich with total POI count per category
        self.site_enriched_data = self.site_enriched_data.merge(
            self.poi_category_counts(), on="site_id", how="left"
        )
        self.site_enriched_data = self.site_enriched_data.fillna(0)

        ## Enrich with closest POI category
        self.site_enriched_data = self.site_enriched_data.merge(
            self.closest_poi_category(), on="site_id", how="left"
        )

        ##
        poi_features = self.extract_poi_data_features()

        poi_features = self.proximity_data.merge(
            poi_features, on=["poi_id", "category"], how="left"
        )

        self.site_enriched_data = self.site_enriched_data.merge(
            poi_features[
                [
                    "site_id",
                    "has_toilet",
                    "has_wifi",
                    "has_seating",
                    "open_24hr",
                    "tags.brand",
                ]
            ],
            on="site_id",
            how="left",
        )
        return self.site_enriched_data
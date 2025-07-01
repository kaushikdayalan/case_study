import requests
from typing import Dict, List


class OverPassAPI:
    def __init__(self):
        self.overpass_url = "https://overpass-api.de/api/interpreter"

    def build_bbox_query(self, bbox: Dict) -> str:
        return f"""
        [out:json][timeout:30];
        (
        node["shop"~"supermarket|convenience"]({bbox['min_lat']},{bbox['min_lon']},{bbox['max_lat']},{bbox['max_lon']});
        way["shop"~"supermarket|convenience"]({bbox['min_lat']},{bbox['min_lon']},{bbox['max_lat']},{bbox['max_lon']});

        node["amenity"="fast_food"]({bbox['min_lat']},{bbox['min_lon']},{bbox['max_lat']},{bbox['max_lon']});
        way["amenity"="fast_food"]({bbox['min_lat']},{bbox['min_lon']},{bbox['max_lat']},{bbox['max_lon']});

        node["amenity"="fuel"]({bbox['min_lat']},{bbox['min_lon']},{bbox['max_lat']},{bbox['max_lon']});
        way["amenity"="fuel"]({bbox['min_lat']},{bbox['min_lon']},{bbox['max_lat']},{bbox['max_lon']});
        );
        out center;
        """

    def build_around_query(
        self, lat: float, lon: float, radius: int = 100
    ) -> str:
        return f"""
        [out:json][timeout:30];
        (
        node["shop"~"supermarket|convenience"](around:{radius},{lat},{lon});
        way["shop"~"supermarket|convenience"](around:{radius},{lat},{lon});

        node["amenity"="fast_food"](around:{radius},{lat},{lon});
        way["amenity"="fast_food"](around:{radius},{lat},{lon});

        node["amenity"="fuel"](around:{radius},{lat},{lon});
        way["amenity"="fuel"](around:{radius},{lat},{lon});
        );
        out center;
        """

    def query_overpass(self, query: str) -> List:
        response = requests.post(self.overpass_url, data={"data": query})
        if response.status_code == 200:
            return response.json()["elements"]
        else:
            print(
                f"Failed to get POI data. {response.status_code}\n{response.json()}"
            )
            return []

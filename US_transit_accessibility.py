import os
import subprocess
import sys
import tempfile
import json
import warnings
from tqdm import tqdm
import api_keys
from pyGTFSHandler.downloaders.mobility_database import MobilityDatabaseClient
import pyGTFSHandler.gtfs_checker as gtfs_checker
from datetime import date, time
import geopandas as gpd
import pandas as pd

refresh_token = api_keys.MOBILITY_DATABASE
orig_gtfs_path = "data/orig_gtfs_files"
gtfs_path = "data/gtfs_files"
osm_path = "data/osm"
output_path = "data/accessibility"

start_time = time(hour=6)
end_time = time(hour=22)
date_type = "businessday"

min_edge_length = 30
distance_steps = [100, 200, 400, 600, 800, 1000, 1250, 1500, 2000]
# interval_steps = [5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 360, 720, 1440]
interval_steps = [5, 7, 10, 15, 20, 30, 45, 60, 90, 120, 180, 360, 720, 1440]

start_date = date(day=1, month=12, year=2025)
end_date = date(day=28, month=2, year=2026)
stop_group_distance = 100
route_type_mapping = {
    "bus": [3, -1],
    "other": [4, 5, 6, 7],
    "tram": [0],
    "subway": [1],
    "rail": [2],
}

# -1 - None
# 0 - tram
# 1 - subway
# 2 - rail
# 3 - bus
# 4 - ferry
# 5 - cable car
# 6 - gondola
# 7 - funicular

route_speed_mapping = [0, 10, 15, 20, 25, 30, 50, 75, 100, 150]
time_step_speeds = 15
speed_direction = "both"
overwrite = False
check_files = True
do_h3 = True
h3_resolution = 11

os.makedirs(orig_gtfs_path, exist_ok=True)
os.makedirs(gtfs_path, exist_ok=True)

api = MobilityDatabaseClient(refresh_token)

aois = gpd.read_file("data/cb_2018_us_state_500k/cb_2018_us_state_500k.shp")
aois["filename"] = aois["NAME"].map(gtfs_checker.normalize_string)
aois = aois.dropna(subset=["filename"])

if aois is None:
    all_aois_wkt = None
else:
    all_aois_wkt = str(aois.union_all().wkt)

feeds = api.search_gtfs_feeds(
    country_code="US",
    is_official=None,  # Set to True if you only want official feeds
)

orig_gtfs_files = api.download_feeds(
    feeds=feeds, download_folder=orig_gtfs_path, overwrite=overwrite
)


stop_intervals_paths = []

for file in tqdm(orig_gtfs_files, desc="Processing GTFS files"):
    filename = os.path.splitext(os.path.basename(file))[0]
    stop_intervals_path = os.path.join(gtfs_path, filename, "stop_intervals.gpkg")

    # Skip if already processed
    if (not overwrite) and os.path.isfile(stop_intervals_path):
        stop_intervals_paths.append(stop_intervals_path)
        continue

    # Create a JSON file with all parameters
    params = {
        "orig_file": file,
        "processed_gtfs_folder": gtfs_path,
        "start_time": start_time.strftime("%H:%M:%S"),
        "end_time": end_time.strftime("%H:%M:%S"),
        "start_date": start_date.strftime("%Y-%m-%d") if start_date else "",
        "end_date": end_date.strftime("%Y-%m-%d") if end_date else "",
        "date_type": date_type,
        "stop_group_distance": stop_group_distance,
        "route_type_mapping": route_type_mapping,
        "route_speed_mapping": route_speed_mapping,
        "time_step_speeds": time_step_speeds,
        "speed_direction": speed_direction,
        "aoi": all_aois_wkt,
    }

    if not check_files:
        params["fast_check"] = True

    if overwrite:
        params["overwrite"] = True

    # Write parameters to a temporary JSON file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        json.dump(params, f)
        params_file = f.name

    # Call stop_quality.py with only the JSON file
    process = [
        sys.executable,
        "-m",
        "pyGTFSHandler.scripts.stop_quality",
        "--params_file",
        params_file,
    ]

    try:
        result = subprocess.run(process, check=True, capture_output=True, text=True)
        stop_intervals_paths.append(stop_intervals_path)
    except subprocess.CalledProcessError as e:
        warnings.warn(
            f"Error processing {file}\n"
            f"Return code: {e.returncode}\n"
            f"stderr:\n{e.stderr}"
        )
    except Exception as e:
        warnings.warn(f"Unexpected error processing {file}: {e}")
    finally:
        # Delete the temporary JSON file to avoid clutter
        os.remove(params_file)

for i in tqdm(range(len(aois)), desc="Processing walksheds"):
    aoi = gpd.GeoDataFrame(geometry=[aois.loc[i, "geometry"]], crs=aois.crs)
    aoi_download = aoi.to_crs(aoi.estimate_utm_crs())
    aoi_download.geometry = aoi_download.geometry.buffer(max(distance_steps))

    aoi_name = str(aois.loc[i, "filename"])
    aoi_path = os.path.join(output_path, aoi_name)
    if not os.path.isdir(aoi_path):
        os.makedirs(aoi_path)

    stop_intervals_path = os.path.join(aoi_path, "stop_intervals.gpkg")
    street_edges_path = os.path.join(aoi_path, "street_edges.gpkg")
    street_nodes_path = os.path.join(aoi_path, "street_nodes.gpkg")
    accessibility_edges_path = os.path.join(aoi_path, "level_of_service_streets.gpkg")
    h3_path = os.path.join(aoi_path, "h3.csv")

    if overwrite or (not os.path.isfile(stop_intervals_path)):
        stop_intervals = []
        for f in stop_intervals_paths:
            stop_intervals.append(
                gpd.read_file(f, bbox=tuple(aoi_download.to_crs(4326).total_bounds))
            )

        if len(stop_intervals) == 0:
            print(f"No pois in {aoi_name}")
            continue

        stop_intervals = pd.concat(stop_intervals)

        if len(stop_intervals) == 0:
            print(f"No pois in {aoi_name}")
            continue

        stop_intervals = stop_intervals[
            stop_intervals.intersects(aoi.to_crs(stop_intervals.crs).union_all())
        ]
        stop_intervals["interval_class"] = interval_steps[-1]
        for j in reversed(interval_steps):
            stop_intervals.loc[
                stop_intervals["mean_interval"] <= j, "interval_class"
            ] = j

        # Get the index of the max min_speed per group
        idx = stop_intervals.groupby(
            ["stop_lat", "stop_lon", "route_type_simple", "interval_class"], sort=False
        )["min_speed"].idxmax()
        idx = idx.dropna().astype(int)
        # Select rows directly
        stop_intervals = stop_intervals.loc[idx].reset_index(drop=True)
        stop_intervals["lat_r"] = stop_intervals.stop_lat.round(5)
        stop_intervals["lon_r"] = stop_intervals.stop_lon.round(5)
        # Get the index of the max min_speed per group
        idx = stop_intervals.groupby(
            ["lat_r", "lon_r", "route_type_simple", "min_speed"], sort=False
        )["interval_class"].idxmin()
        idx = idx.dropna().astype(int)
        # Select rows directly
        stop_intervals = stop_intervals.drop(columns=["lat_r", "lon_r"])
        stop_intervals = stop_intervals.loc[idx].reset_index(drop=True)

        stop_intervals.to_file(stop_intervals_path)

    if (
        overwrite
        or (not os.path.isfile(accessibility_edges_path))
        or (do_h3 and (not os.path.isfile(h3_path)))
    ):
        params = {
            "poi_file": stop_intervals_path,
            "poi_quality_column": ["interval_class", "route_type_simple", "min_speed"],
            "output_path": aoi_path,
            "street_path": osm_path,
            "aoi": str(aoi.to_crs(4326).union_all().wkt),
            "min_edge_length": min_edge_length,
            "h3_resolution": h3_resolution,
            "distance_steps": distance_steps,
        }

        if do_h3:
            params["h3"] = True

        if overwrite:
            params["overwrite"] = True

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            json.dump(params, f)
            params_file = f.name

        process = [
            sys.executable,
            "-m",
            "UrbanAccessAnalyzer.scripts.walksheds",
            "--params_file",
            params_file,
        ]

        # Try running the process
        try:
            result = subprocess.run(process, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            warnings.warn(
                f"Error processing {aoi_name}\n"
                f"Return code: {e.returncode}\n"
                f"stderr:\n{e.stderr}"
            )
        except Exception as e:
            warnings.warn(f"Unexpected error processing {aoi_name}: {e}")
        finally:
            os.remove(params_file)

import copy
import sys
from typing import Union, List, Optional, Dict, Any

import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union

import pygris
from pygris.utils import erase_water
from pygris.data import get_census
import us  # For state normalization

import api_keys

CENSUS_LATEST_YEARS: Dict[str, int] = {
    "dec/dhc": 2020,  # Available for blocks
    "acs/acs5": 2023,  # Only block groups and higher
    "acs/acs1": 2024,  # Only places and higher
}

census_api_key: str = api_keys.US_CENSUS


# ----------------------------
# Utilities
# ----------------------------


def _to_list(x: Any) -> List:
    """
    Convert a parameter to a list.

    Parameters
    ----------
    x : any or list
        Parameter that is either a single element or a list.

    Returns
    -------
    list
        Parameter as a list.
    """
    return x if isinstance(x, list) else [x]


def format_filter(
    filter: Dict[str, Union[str, List[str]]], inplace: bool = False
) -> Dict[str, List[str]]:
    """
    Normalize filter values for consistent matching.

    - 'state' -> USPS abbreviation (case-insensitive, accepts full name or abbrev)
    - other keys (county, place) -> lowercase for case-insensitive matching

    Parameters
    ----------
    filter : dict
        Filter dictionary with keys like 'state', 'county', 'place'.
    inplace : bool, default=False
        If True, modifies the input dictionary in place. Otherwise returns a new normalized dictionary.

    Returns
    -------
    dict
        Normalized filter dictionary with lists of strings.
    """
    target = filter if inplace else {}

    for key, val in filter.items():
        vals = [val] if isinstance(val, str) else val
        if key.lower() == "state":
            target["state"] = [us.states.lookup(v.strip()).abbr for v in vals]
        else:
            target[key.lower()] = [v.strip().lower() for v in vals]

    return target


def format_categories_dict(
    categories: Dict[str, Dict], inplace: bool = False
) -> Dict[str, Dict]:
    """
    Format a categories dictionary for census data requests.

    Parameters
    ----------
    categories : dict
        Dictionary of census categories.
    inplace : bool, default=False
        If True, modifies the original dictionary; otherwise returns a formatted copy.

    Returns
    -------
    dict
        Formatted categories dictionary with standardized sources, years, and field names.
    """
    if not inplace:
        categories = copy.deepcopy(categories)

    def source_to_api_dir(source: str) -> str:
        """Convert source name to API-compatible string."""
        if source.startswith("acs"):
            return "acs/acs1" if source == "acs1" else "acs/acs5"
        elif source.startswith("dec"):
            dec_suffix = source.split("_")[-1] if "_" in source else "dhc"
            return f"dec/{dec_suffix}"
        elif source == "dhc":
            return "dec/dhc"
        else:
            print(
                f"Warning: Unrecognized source '{source}'. Returning source as is.",
                file=sys.stderr,
            )
            return source

    def format_years(source: str, years: Optional[Union[int, List[int]]]) -> List[int]:
        """Ensure years are a list and defaults to latest available."""
        if years is None:
            return _to_list(CENSUS_LATEST_YEARS[source])
        return _to_list(years)

    # Standardize source and years
    for cat_name, cat_dict in categories.items():
        cat_dict["source"] = source_to_api_dir(cat_dict["source"])
        cat_dict["years"] = format_years(
            cat_dict["source"], cat_dict.get("years", None)
        )

    # Determine if years should be prefixed in field names
    years_by_source: Dict[str, set] = {}
    for cat_name, cat_dict in categories.items():
        source, years = cat_dict["source"], cat_dict["years"]
        years_by_source.setdefault(source, set()).update(years)
    years_as_prefix = any(len(years) > 1 for years in years_by_source.values())

    def get_field_name(
        cat_name: str, field_name: str, year: Optional[int] = None
    ) -> str:
        """Generate standardized field name."""
        return (
            f"{year}_{cat_name}_{field_name}"
            if years_as_prefix and year
            else f"{cat_name}_{field_name}"
        )

    # Reformat fields
    for cat_name, cat_dict in categories.items():
        fields_formatted: Dict[str, Any] = {}
        fields_universe_formatted: Dict[str, Any] = {}
        if "default" in cat_dict.get("fields_universe", {}):
            fields_universe_formatted["default"] = cat_dict["fields_universe"][
                "default"
            ]

        for field_name, field_codes in cat_dict["fields"].items():
            for year in cat_dict["years"]:
                new_field_name = get_field_name(cat_name, field_name, year=year)
                fields_formatted[new_field_name] = field_codes
                if field_name in cat_dict.get("fields_universe", {}):
                    fields_universe_formatted[new_field_name] = cat_dict[
                        "fields_universe"
                    ][field_name]

        cat_dict["fields"] = fields_formatted
        cat_dict["fields_universe"] = fields_universe_formatted

    return categories


# ----------------------------
# Geospatial Functions
# ----------------------------


def load_shapes(
    filter: Dict[str, Union[str, List[str]]] = {},
    aoi: Optional[Union[gpd.GeoDataFrame, gpd.GeoSeries]] = None,
    level: str = "block",
    year: Optional[int] = None,
    remove_water: bool = True,
    epsg: int = 3857,
    cache: bool = True,
) -> gpd.GeoDataFrame:
    """
    Load US Census shapes at various geographic levels, filtered by state, county, place, or AOI.
    Automatically uses fast filters where available and infers counties from place geometry
    for block, blockgroup, and tract levels to ensure speed.

    Parameters
    ----------
    filter : dict, optional
        Keys can include:
        - 'state': str or list of state abbreviations
        - 'county': str or list of county names
        - 'place': str or list of place names
        Example: {'state':'MA', 'county':['Suffolk'], 'place':'Boston'}
        If 'state' is omitted, the function will search all states.
    aoi : GeoDataFrame or GeoSeries, optional
        Area of interest. If provided, shapes will be clipped to this AOI.
    level : str
        Geography level: 'block', 'blockgroup', 'tract', 'place', 'county'
    year : int, optional
        Year of the TIGER/ACS data.
    remove_water : bool, default True
        Remove water areas for block/blockgroup/tract levels.
    epsg : int, default 3857
        Output CRS.
    cache : bool, default True
        Whether to cache downloaded TIGER files.

    Returns
    -------
    GeoDataFrame
        Filtered US Census geometries.
    """
    filter = format_filter(filter)

    # Prepare geography functions
    geometry_funcs = {
        "block": pygris.blocks,
        "blockgroup": pygris.block_groups,
        "block group": pygris.block_groups,
        "block_group": pygris.block_groups,
        "tract": pygris.tracts,
        "place": pygris.places,
        "county": pygris.counties,
    }
    for k, v in list(geometry_funcs.items()):
        geometry_funcs[k + "s"] = v  # allow plural

    # Determine states
    states = _to_list(filter.get("state", None))
    if states is None and "place" in filter:
        states = pygris.states(year=year).STUSPS.tolist()

    dfs = []

    for state in states:
        subset_geom = None
        county_fips = None

        # AOI geometry
        if aoi is not None:
            aoi_proj = aoi.to_crs(epsg)
            subset_geom = unary_union(aoi_proj.geometry)

        # Place filter
        if "place" in filter:
            places_list = _to_list(filter["place"])
            places_df = pygris.places(state=state, year=year or 2024, cache=cache)
            selected_places = places_df[
                places_df["NAME"].str.lower().isin([p.lower() for p in places_list])
            ]
            if selected_places.empty:
                continue  # Place not in this state
            place_geom = unary_union(selected_places.geometry)
            subset_geom = (
                place_geom
                if subset_geom is None
                else unary_union([subset_geom, place_geom])
            )

            # Infer counties intersecting place (for fast download)
            if level in ["block", "blockgroup", "tract"]:
                counties_df = pygris.counties(
                    state=state, year=year or 2024, cache=cache
                )
                intersecting_counties = counties_df[counties_df.intersects(place_geom)]
                if not intersecting_counties.empty:
                    county_fips = intersecting_counties["COUNTYFP"].tolist()

        # County filter
        if "county" in filter:
            counties_list = _to_list(filter["county"])
            counties_df = pygris.counties(state=state, year=year or 2024, cache=cache)
            selected_counties = counties_df[
                counties_df["NAME"].str.lower().isin([c.lower() for c in counties_list])
            ]
            if selected_counties.empty:
                continue  # No matching counties in this state
            county_geom = unary_union(selected_counties.geometry)
            subset_geom = (
                county_geom
                if subset_geom is None
                else unary_union([subset_geom, county_geom])
            )
            county_fips = (
                selected_counties["COUNTYFP"].tolist()
                if county_fips is None
                else list(
                    set(county_fips) & set(selected_counties["COUNTYFP"].tolist())
                )
            )

        # Prepare pygris arguments
        load_kwargs = {"state": state, "year": year or 2024, "cache": cache}
        if county_fips and level in ["block", "blockgroup", "tract"]:
            load_kwargs["county"] = county_fips

        # subset_by only if no county fast filter
        if subset_geom is not None and not (
            county_fips and level in ["block", "blockgroup", "tract"]
        ):
            load_kwargs["subset_by"] = gpd.GeoSeries([subset_geom], crs=epsg)

        # Load shapes
        shapes = geometry_funcs[level](**load_kwargs)

        # Remove water if needed
        if (
            remove_water
            and level in ["block", "blockgroup", "tract"]
            and len(shapes) > 0
        ):
            required_fips_cols = ["STATEFP", "COUNTYFP"]

            # Ensure FIPS columns exist (spatial join with counties if missing)
            missing_cols = [
                col for col in required_fips_cols if col not in shapes.columns
            ]
            if missing_cols:
                counties_df = pygris.counties(
                    state=state, year=year or 2024, cache=cache
                )
                shapes = gpd.sjoin(
                    shapes,
                    counties_df[["geometry", "STATEFP", "COUNTYFP"]],
                    how="left",
                    predicate="intersects",
                )
                # Drop duplicate geometry column from join
                shapes = shapes.drop(
                    columns=[col for col in shapes.columns if col.endswith("_right")],
                    errors="ignore",
                )

            # Only call erase_water if shapes intersect at least one county
            counties_df = pygris.counties(state=state, year=year or 2024, cache=cache)

            # Reproject counties to match shapes CRS
            if counties_df.crs != shapes.crs:
                counties_df = counties_df.to_crs(shapes.crs)

            intersecting_counties = counties_df[
                counties_df.intersects(shapes.union_all())
            ]
            if not intersecting_counties.empty and all(
                col in shapes.columns for col in required_fips_cols
            ):
                target_year = 2024 if (year is None or year >= 2025) else year
                shapes = erase_water(shapes, year=target_year)
            else:
                print(
                    f"Warning: Shapes do not intersect any counties. Skipping erase_water for {level}.",
                    file=sys.stderr,
                )

        # Reproject
        if epsg:
            shapes = shapes.to_crs(epsg=epsg)

        dfs.append(shapes)

    if not dfs:
        return gpd.GeoDataFrame(columns=[], crs=f"EPSG:{epsg}")

    return gpd.GeoDataFrame(pd.concat(dfs, ignore_index=True), crs=f"EPSG:{epsg}")


# ----------------------------
# Census Tabular Functions
# ----------------------------


def load_fields(
    categories: Dict[str, Dict],
    api_key: str = census_api_key,
    state: Union[str, us.states.State] = "MA",
    level: str = "blockgroup",
    compute_ratios: bool = True,
    add_place_names: bool = False,
) -> pd.DataFrame:
    """
    Pull tabular census data from Decennial Census or ACS and compute ratios.

    Parameters
    ----------
    categories : dict
        Categories dict formatted for census variables.
    api_key : str
        Census API key.
    state : str or us.states.State
        State to query.
    level : str
        Geography level.
    compute_ratios : bool, default=True
        Whether to compute ratio fields.
    add_place_names : bool, default=False
        Include place names in output.

    Returns
    -------
    DataFrame
        Tabular census data with optional ratio fields.
    """
    if isinstance(state, str):
        state = us.states.lookup(state)
    state_id = state.fips

    categories = format_categories_dict(categories, inplace=False)
    # Filter categories based on geography level
    if level == "block":
        # Only decennial sources are supported at block level
        categories = {
            k: v for k, v in categories.items() if v["source"].startswith("dec")
        }
    else:
        # All sources allowed for block group and above
        categories = {k: v for k, v in categories.items()}

    # Prepare field definitions
    fields_list = []
    for cat_name, cat_dict in categories.items():
        for field_name, field_codes in cat_dict["fields"].items():
            for year in cat_dict["years"]:
                fields_list.append(
                    {
                        "name": field_name,
                        "source": cat_dict["source"],
                        "year": year,
                        "sum_codes": field_codes,
                        "universe_code": cat_dict["fields_universe"].get(
                            field_name, cat_dict["fields_universe"]["default"]
                        ),
                    }
                )

    # Add 'E' suffix for ACS fields
    def add_e(c: str) -> str:
        if c is None or c in ["DENSITY_ONLY", "NO_DENSITY_OR_RATIO"]:
            return c
        return c if c.endswith("E") else c + "E"

    for field in fields_list:
        if field["source"].startswith("acs"):
            field["sum_codes"] = [add_e(c) for c in _to_list(field["sum_codes"])]
            field["universe_code"] = add_e(field["universe_code"])

    # Aggregate codes by source-year
    codes_by_source_year: Dict[tuple, set] = {}
    for field in fields_list:
        key = (field["source"], field["year"])
        codes_by_source_year.setdefault(key, set()).update(field["sum_codes"])
        if field["universe_code"] and field["universe_code"] not in [
            "DENSITY_ONLY",
            "NO_DENSITY_OR_RATIO",
        ]:
            codes_by_source_year[key].add(field["universe_code"])
        if add_place_names and level.startswith("place"):
            codes_by_source_year[key].add("NAME")

    # Fetch census data
    GEO_HIERARCHIES = {
        "block": ["state", "county", "tract", "block"],
        "blockgroup": ["state", "county", "tract", "block group"],
        "tract": ["state", "county", "tract"],
        "place": ["state", "place"],
    }
    for k, v in list(GEO_HIERARCHIES.items()):
        GEO_HIERARCHIES[k + "s"] = v  # allow plural

    df_by_source_year: Dict[tuple, pd.DataFrame] = {}
    for (source, year), all_fields in codes_by_source_year.items():
        geo_hierarchy = GEO_HIERARCHIES[level]
        geo_for = geo_hierarchy[-1] + ":*"
        geo_in = [
            f"state:{state_id}" if g == "state" else f"{g}:*"
            for g in geo_hierarchy[:-1]
        ]

        df_by_source_year[(source, year)] = pd.DataFrame(
            get_census(
                dataset=source,
                year=year,
                variables=list(all_fields),
                params={"for": geo_for, "in": geo_in, "key": api_key},
                return_geoid=True,
                guess_dtypes=True,
            )
        )

    # Process fields
    df_processed_by_source_year: Dict[tuple, pd.DataFrame] = {}
    final_df_fields: List[str] = []
    final_df_fields_set: set = set()

    def add_fieldname(field_name: str, prepend: bool = False):
        if field_name not in final_df_fields_set:
            if prepend:
                final_df_fields.insert(0, field_name)
            else:
                final_df_fields.append(field_name)
            final_df_fields_set.add(field_name)

    for field in fields_list:
        add_fieldname(field["name"])
        source, year = field["source"], field["year"]
        df_raw = df_by_source_year[(source, year)]
        df_proc = df_processed_by_source_year.get((source, year), pd.DataFrame())

        for col in ["NAME", "GEOID"]:
            if col in df_raw.columns and col not in df_proc.columns:
                df_proc[col] = df_raw[col]
                add_fieldname(col, prepend=True)

        df_proc[field["name"]] = df_raw[field["sum_codes"]].sum(axis=1)

        if (
            compute_ratios
            and field["universe_code"]
            and field["universe_code"] not in ["DENSITY_ONLY", "NO_DENSITY_OR_RATIO"]
        ):
            ratio_entry_name = f"{field['name']}_ratio"
            df_proc[ratio_entry_name] = (
                df_proc[field["name"]] / df_raw[field["universe_code"]]
            )
            add_fieldname(ratio_entry_name)

        df_processed_by_source_year[(source, year)] = df_proc

    # Merge data from all source-year combinations
    join_on = ["GEOID"] + (["NAME"] if "NAME" in final_df_fields_set else [])
    df_final: Optional[pd.DataFrame] = None
    for df_proc in df_processed_by_source_year.values():
        df_final = (
            df_proc
            if df_final is None
            else df_final.merge(df_proc, on=join_on, how="outer")
        )

    # Rename GEOID for blocks
    if level == "block":
        # Take the year from the first field (all fields should have same year for decennial blocks)
        block_year = next(iter(df_processed_by_source_year.keys()))[1]
        df_final = df_final.rename(columns={"GEOID": f"GEOID{str(block_year)[-2:]}"})
        final_df_fields = [
            f"GEOID{year}" if i == "GEOID" else i for i in final_df_fields
        ]

    return df_final[final_df_fields]


# ----------------------------
# Join Geospatial and Census
# ----------------------------


def join_census_and_add_densities(
    df_geo: gpd.GeoDataFrame,
    df_census: pd.DataFrame,
    density_fields: Optional[List[str]] = None,
    categories: Optional[Dict[str, Dict]] = None,
) -> gpd.GeoDataFrame:
    """
    Join census tabular data with geospatial shapes and compute density fields.

    Parameters
    ----------
    df_geo : GeoDataFrame
        Geospatial data with geometries.
    df_census : DataFrame
        Census tabular data.
    density_fields : list of str, optional
        Fields for which to compute densities.
    categories : dict, optional
        Categories dictionary to infer density fields if not provided.

    Returns
    -------
    GeoDataFrame
        Geospatial data joined with census data and densities.
    """
    # Automatically detect GEOID column in df_census
    geoid_col = next((c for c in df_census.columns if "GEOID" in c.upper()), None)
    if geoid_col is None:
        raise KeyError("No column containing 'GEOID' found in df_census.")

    # Determine if NAME should also be used for joining
    join_on = [geoid_col] + (
        ["NAME"] if "NAME" in df_census.columns and "NAME" in df_geo.columns else []
    )

    # Merge
    df_geodata = df_geo.merge(df_census, on=join_on, how="outer")

    # Determine density fields if not provided
    if density_fields is None and categories is not None:
        density_fields = []
        categories = format_categories_dict(categories, inplace=False)
        for cat_name, cat_dict in categories.items():
            for field_name in cat_dict["fields"].keys():
                universe_code = cat_dict["fields_universe"].get(
                    field_name, cat_dict["fields_universe"]["default"]
                )
                if universe_code and universe_code != "NO_DENSITY_OR_RATIO":
                    density_fields.append(field_name)
        density_fields = [f for f in density_fields if f in df_geodata.columns]

    # Compute densities
    final_df_fields: List[str] = []
    density_fields_set = set(density_fields or [])
    for col in df_geodata.columns:
        final_df_fields.append(col)
        if col in density_fields_set:
            density_field_name = f"{col}_density"
            df_geodata[density_field_name] = df_geodata[col] / df_geodata.geometry.area
            final_df_fields.append(density_field_name)

    return df_geodata[final_df_fields]

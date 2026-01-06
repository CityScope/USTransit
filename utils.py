import math
import re
from itertools import product
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    TypeVar,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
import polars as pl

import warnings

# Type variables
T = TypeVar("T")  # Generic input type (interval, route_type, etc.)
Q = TypeVar("Q", bound=float)  # Quality value type


def parse_column_with_pattern(column_name: str, column_pattern: str) -> Dict[str, Any]:
    """
    Extract parameters from a column name using a format-style pattern.

    Parameters
    ----------
    column_name : str
        The actual column name to parse.
    column_pattern : str
        Format string containing placeholders, e.g.,
        "interval_class_{interval}_-_route_type_simple_{route_type}_-_min_speed_{speed}".

    Returns
    -------
    Dict[str, Any]
        Dictionary mapping parameter names to values, converting numeric-looking
        strings to int or float.

    Raises
    ------
    ValueError
        If the column_name does not match the pattern.
    """
    # Escape all fixed parts
    regex_pattern = re.escape(column_pattern)

    # Extract placeholder names
    param_names = re.findall(r"{(\w+)}", column_pattern)
    if not param_names:
        raise ValueError("No parameters found in column_pattern")

    # Replace placeholders with named capture groups
    for i, name in enumerate(param_names):
        if i == len(param_names) - 1:
            # Last parameter: greedy to match everything until end
            regex_pattern = regex_pattern.replace(
                r"\{" + name + r"\}", rf"(?P<{name}>.+)"
            )
        else:
            # Non-greedy for other parameters
            regex_pattern = regex_pattern.replace(
                r"\{" + name + r"\}", rf"(?P<{name}>.+?)"
            )

    match = re.match(regex_pattern, column_name)
    if not match:
        raise ValueError(
            f"Column '{column_name}' does not match pattern '{column_pattern}'"
        )

    params = match.groupdict()

    # Convert numeric-looking values
    for k, v in params.items():
        try:
            if "." in v:
                params[k] = float(v)
            else:
                params[k] = int(v)
        except ValueError:
            params[k] = v  # leave as string

    return params


def elasticity_based_quality(
    value: float,
    reference: float,
    elasticity: Union[float, Callable[[float], float], List[Sequence[float]]],
    steps: int = 200,
) -> float:
    """
    Compute quality using elasticity-based integration.

    Parameters
    ----------
    value : float
        The current value of the variable.
    reference : float
        Reference value (e.g., baseline).
    elasticity : float, callable, or list of [lower_bound, elasticity]
        - If float: constant elasticity.
        - If callable: function of x returning elasticity.
        - If list: piecewise elasticity [[lower_bound, e], ...] applied for x >= lower_bound.
    steps : int, default=200
        Number of steps for numerical integration.

    Returns
    -------
    float
        Quality value in (0, 1], decreasing as value moves away from reference.
    """
    # Build elasticity function
    if isinstance(elasticity, (int, float)):

        def elasticity_fn(x: float) -> float:
            return elasticity

    elif isinstance(elasticity, list):
        # Piecewise elasticity
        processed = [(-math.inf if lb is None else lb, e) for lb, e in elasticity]
        processed.sort(key=lambda t: t[0])

        def elasticity_fn(x: float) -> float:
            current_e = processed[0][1]
            for lb, e in processed:
                if x >= lb:
                    current_e = e
                else:
                    break
            return current_e

    elif callable(elasticity):
        elasticity_fn = elasticity

    else:
        raise TypeError(
            "elasticity must be a float, a callable, or a list of [lower_bound, elasticity]"
        )

    # If value equals reference, quality is 1
    if value == reference:
        return 1.0

    xs = np.linspace(reference, value, steps)
    integrand = [elasticity_fn(x) / x for x in xs]
    integral = np.trapezoid(integrand, xs)
    return math.exp(-integral)


def calibrate_quality_func(
    quality_func: Callable[..., float],
    *,
    min_quality: float = 0.1,
    max_quality: float = 1.0,
    min_point: Optional[Sequence[T]] = None,
    max_point: Optional[Sequence[T]] = None,
    variable_steps: Optional[List[Any]] = None,
) -> Callable[..., float]:
    """
    Normalize a multi-parameter quality function to a given range.

    Parameters
    ----------
    quality_func : callable
        Function accepting positional arguments (e.g., interval, route_type, speed, distance).
    min_quality : float, default=0.1
        Minimum normalized quality.
    max_quality : float, default=1.0
        Maximum normalized quality.
    min_point : sequence, optional
        Explicit point to define minimum quality.
    max_point : sequence, optional
        Explicit point to define maximum quality.
    variable_steps : list of iterables, optional
        Steps for each argument to generate combinations if min/max points are not provided.

    Returns
    -------
    callable
        Function with same arguments as `quality_func` that returns normalized quality.
    """
    combinations: List[Tuple[T, ...]] = []

    # Generate combinations if needed
    if (min_point is None or max_point is None) and variable_steps is not None:
        steps = [
            sorted(step)
            if isinstance(step, Iterable) and not isinstance(step, (str, bytes))
            else [step]
            for step in variable_steps
        ]
        combinations.extend(product(*steps))

    if min_point is not None:
        combinations.append(tuple(min_point))
    if max_point is not None:
        combinations.append(tuple(max_point))

    if not combinations:
        raise ValueError("No points provided to compute quality range")

    qualities = [quality_func(*p) for p in combinations]

    q_min = quality_func(*min_point) if min_point is not None else min(qualities)
    q_max = quality_func(*max_point) if max_point is not None else max(qualities)

    if q_min == q_max:
        raise ValueError("q_min and q_max are equal; cannot normalize")

    def access_quality(*args: T) -> float:
        x = quality_func(*args)
        return min_quality + (x - q_min) * (max_quality - min_quality) / (q_max - q_min)

    return access_quality


def generate_access_df(
    access_quality_func: Callable[..., float],
    variable_steps: Union[List[Any], Dict[str, Any]],
    *,
    column_pattern: Optional[str] = None,
    access_qualities: Optional[Iterable[float]] = None,
) -> pd.DataFrame:
    """
    Generate a DataFrame of access values for all parameter combinations.

    Parameters
    ----------
    access_quality_func : callable
        Function accepting positional arguments, returning a float.
    variable_steps : list of iterables or dict of iterables
        Steps for each argument or named dict of steps. If using list the last iterable is 'distance' if len mismatch with column_pattern.
    column_pattern : str, optional
        Format string to generate column names. Defaults to "{col0}_{col1}_...".
    access_qualities : iterable of float, optional
        Values to round access values to. If None, no rounding is applied.

    Returns
    -------
    pd.DataFrame
        Columns include parameter values, 'access', 'access_rounded', 'rounding_error', and 'column'.
    """
    # Determine columns
    if isinstance(variable_steps, dict):
        columns = list(variable_steps.keys())
        steps_list = list(variable_steps.values())
    else:
        if column_pattern is None:
            columns = [f"arg{i}" for i in range(len(variable_steps))]
            steps_list = variable_steps
        else:
            columns = list(
                parse_column_with_pattern(column_pattern, column_pattern).keys()
            )
            steps_list = variable_steps
            if len(columns) != len(steps_list):
                columns.append("distance")

            if len(columns) != len(steps_list):
                warnings.warn(
                    f"Length mismatch: variable_steps has length {len(variable_steps)} "
                    f"but column_pattern defines {len(columns)} columns {column_pattern} columns {columns}.",
                    UserWarning,
                    stacklevel=2,
                )
                columns = [f"arg{i}" for i in range(len(variable_steps))]
                steps_list = variable_steps

    steps_list = [
        sorted(step)
        if isinstance(step, Iterable) and not isinstance(step, (str, bytes))
        else [step]
        for step in steps_list
    ]

    # Generate all combinations
    combinations = list(product(*steps_list))
    df = pd.DataFrame(combinations, columns=columns)

    # Compute access
    df["access"] = df.apply(
        lambda row: access_quality_func(*[row[c] for c in columns]), axis=1
    )

    # Rounding if provided
    if access_qualities is not None:
        access_arr = np.array(access_qualities)

        def round_to_nearest(x):
            idx = np.abs(access_arr - x).argmin()
            return access_arr[idx]

        df["access_rounded"] = df["access"].apply(round_to_nearest)
        df["rounding_error"] = np.abs(df["access"] - df["access_rounded"])
    else:
        df["access_rounded"] = df["access"]
        df["rounding_error"] = 0.0

    # Generate column names
    if column_pattern is None:
        column_pattern = "_".join(f"{{{col}}}" for col in columns)

    def generate_column(row):
        try:
            return column_pattern.format(**{c: row[c] for c in columns})
        except KeyError:
            return "_".join(str(row[c]) for c in columns)

    df["column"] = df.apply(generate_column, axis=1)

    if "distance" in df.columns:
        # Sort and deduplicate
        df = df.sort_values(["column", "access_rounded", "distance"], ascending=False)
    else:
        df = df.sort_values(["column", "access_rounded"], ascending=False)

    df = df.drop_duplicates(["column", "access_rounded"], keep="first")

    return df.reset_index(drop=True)


def assign_access_value(
    lf: Union[pl.LazyFrame, pl.DataFrame],
    access_quality_func: Callable[..., float],
    column_pattern,
    distance_steps: Optional[Sequence[float]] = None,
) -> pl.LazyFrame:
    """
    Assign access values to Polars columns based on column_pattern.

    Parameters
    ----------
    lf : pl.LazyFrame or pl.DataFrame
        Input table.
    access_quality_func : callable
        Function accepting positional arguments, returning float.
    column_pattern : str, optional
        Format string describing column naming pattern.
    distance_steps : sequence of float, optional
        Distance steps for mapping numeric values.

    Returns
    -------
    pl.LazyFrame
        LazyFrame with transformed access columns and a combined 'access' column.
    """
    # Ensure LazyFrame
    if isinstance(lf, pl.DataFrame):
        lf = lf.lazy()

    # Select columns
    if column_pattern is None:
        columns = lf.collect_schema().names()
    else:
        fixed_parts = re.split(r"{\w+}", column_pattern)
        columns = [
            col
            for col in lf.collect_schema().names()
            if all(part in col for part in fixed_parts)
        ]
    if not columns:
        raise ValueError("No columns found matching the column_pattern")

    lf = lf.with_columns([pl.col(col).cast(float).alias(col) for col in columns])

    transform_columns = []

    for column in columns:
        # Distance steps
        if distance_steps is None:
            col_distance_steps = (
                lf.select(column).drop_nulls().unique().collect()[column]
            )
            col_distance_steps = sorted([float(d) for d in col_distance_steps])
        else:
            col_distance_steps = sorted(np.unique(distance_steps))

        # Extract parameters from column name
        params_dict = parse_column_with_pattern(column, column_pattern)

        variable_steps = [[v] for v in params_dict.values()] + [col_distance_steps]

        # Generate access mapping DataFrame
        access_df = generate_access_df(
            access_quality_func,
            variable_steps=variable_steps,
        )
        # Convert keys to a list to preserve order
        params_keys = list(params_dict.keys())
        rename_map = {f"arg{i}": params_keys[i] for i in range(len(params_keys))}
        # The last argument corresponds to distance
        rename_map[f"arg{len(params_keys)}"] = "distance"
        access_df = access_df.rename(columns=rename_map)
        access_df = access_df.sort_values(
            ["access_rounded", "distance"], ascending=False
        )
        access_df = access_df.drop_duplicates(
            ["access_rounded"], keep="first"
        ).reset_index(drop=True)

        # Build Polars expressions
        mapping = dict(zip(access_df["distance"], access_df["access_rounded"]))
        expr = None
        for d, a in mapping.items():
            if expr is None:
                expr = pl.when(pl.col(column) <= d).then(a)
            else:
                expr.when(pl.col(column) <= d).then(a)

        expr = expr.otherwise(0)

        transform_columns.append(expr.alias(column))

    # Apply transformations
    lf = lf.with_columns(transform_columns)

    # Compute max access across transformed columns
    lf = lf.with_columns(pl.max_horizontal(columns).alias("access")).drop(columns)

    return lf

"""
ExtractDownstreamRiverCourse.py

Given a CSV of dams, snaps each dam to the nearest reach (within a fixed
search radius) in a HydroRIVERS-format river network (HYRIV_ID / NEXT_DOWN /
LENGTH_KM / ENDORHEIC schema, see HydroRIVERS Technical Documentation v1.0)
and walks the NEXT_DOWN chain downstream until it terminates (NEXT_DOWN == 0),
which corresponds to the reach draining into the ocean or into an endorheic
(inland) sink.

Project folder structure:
    Extract_Downstream_River_Course/
        Data/    - input dam CSV (Dam ID, Dam name, Latitude, Longitude)
        Module/  - this script
        Plot/
            <csv_name>/
                <Dam ID>_WaterCourse.gpkg   - one file per dam
        Output/  - QC CSV (created alongside Plot, one level up from Module)

The output subfolder under Plot/ is named automatically after the input CSV
(its filename without extension), and each dam's file is named
"{Dam ID}_WaterCourse.gpkg".

Usage (batch, run from Module/ or anywhere — paths resolve off this file's
location):
    python ExtractDownstreamRiverCourse.py --dams dam_portfolio.csv

    Or simply run with no arguments and it will prompt for the CSV filename
    (and optionally river path / search radius / QC filename, each with a
    sensible default you can just press Enter to accept):
        python ExtractDownstreamRiverCourse.py

    (--dams can be a bare filename, resolved against ../Data, or a full path.
    --rivers defaults to the HydroRIVERS_v10_shp folder in the standard
    Resources location; pass --rivers to override, either a direct .shp path
    or its containing folder.)

    Optional:
        --radius 2000        search radius in meters (default 2000)
        --qc some_name.csv   QC CSV filename (default: {csv_name}_output.csv, saved under Output/)

Usage (single dam, interactive):
    python ExtractDownstreamRiverCourse.py --single
"""

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEARCH_RADIUS_M_DEFAULT = 2000.0   # dam must have a reach within this radius
MAX_CHAIN_STEPS = 200_000          # safety cap against corrupt/cyclic NEXT_DOWN data
SAVE_EVERY = 10                    # progressive QC save cadence, matches other scripts

REQUIRED_COLS = ["HYRIV_ID", "NEXT_DOWN", "LENGTH_KM", "ENDORHEIC"]

# Default HydroRIVERS location (global coverage). Can be overridden with --rivers.
DEFAULT_RIVERS_PATH = (
    "/Users/filou/MyProjects/MyGISProjects/Resources/Maps and data/"
    "HydroRIVERS/HydroRIVERS_v10_shp"
)

# Project root is one level up from this script (Module/ -> project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Data"
PLOT_DIR = PROJECT_ROOT / "Plot"
OUTPUT_DIR = PROJECT_ROOT / "Output"


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def resolve_input_path(path_str, default_dir):
    """Allow either a bare filename (resolved against default_dir) or a full path."""
    p = Path(path_str)
    if p.is_absolute() or p.exists():
        return p
    return default_dir / p


def resolve_rivers_path(path_str):
    """
    Accept either a direct path to a .shp/.gpkg file, or a folder containing
    exactly one .shp file (as with the distributed HydroRIVERS_v10_shp bundle).
    """
    p = Path(path_str)
    if p.is_file():
        return p
    if p.is_dir():
        shp_files = sorted(p.glob("*.shp"))
        if len(shp_files) == 1:
            return shp_files[0]
        if len(shp_files) > 1:
            raise ValueError(
                f"Multiple .shp files found in {p}, specify one directly with --rivers: "
                f"{[f.name for f in shp_files]}"
            )
        raise ValueError(f"No .shp file found in {p}")
    raise ValueError(f"River network path not found: {p}")


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def load_river_network(rivers_path):
    """Load the HydroRIVERS layer once and build the NEXT_DOWN lookup."""
    rivers_path = resolve_rivers_path(rivers_path)
    print(f"Loading river network from {rivers_path}...")
    rivers = gpd.read_file(rivers_path)

    missing = [c for c in REQUIRED_COLS if c not in rivers.columns]
    if missing:
        raise ValueError(
            f"River network is missing expected HydroRIVERS columns: {missing}"
        )

    if rivers.crs is None:
        raise ValueError("River network has no CRS defined.")

    # Metric-CRS copy purely for accurate distance measurement (search radius,
    # snap distance). HydroRIVERS ships in geographic WGS84 (EPSG:4326);
    # degrees are not a reliable distance unit. Output geometry stays in the
    # original CRS.
    rivers_metric = rivers.to_crs("EPSG:4087")  # World Equidistant Cylindrical

    next_down_lookup = rivers.set_index("HYRIV_ID")["NEXT_DOWN"]
    if next_down_lookup.index.duplicated().any():
        dupes = next_down_lookup.index[next_down_lookup.index.duplicated()].unique()
        raise ValueError(f"Duplicate HYRIV_ID values found in river network: {list(dupes)[:5]}...")

    return rivers, rivers_metric, next_down_lookup


def find_nearest_reach_within_radius(rivers, rivers_metric, lat, lon, radius_m):
    """
    Find the HYRIV_ID nearest to a dam point, restricted to reaches whose
    geometry falls within `radius_m` of the point. Returns (HYRIV_ID,
    snap_distance_m) or (None, None) if nothing is found within the radius.
    """
    point_wgs84 = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326")
    point_metric = point_wgs84.to_crs(rivers_metric.crs).iloc[0]

    search_area = point_metric.buffer(radius_m)
    candidate_positions = list(rivers_metric.sindex.query(search_area, predicate="intersects"))

    if not candidate_positions:
        return None, None

    candidates = rivers_metric.iloc[candidate_positions]
    distances = candidates.geometry.distance(point_metric)
    best_pos_in_candidates = distances.values.argmin()
    best_position = candidate_positions[best_pos_in_candidates]

    nearest_reach = rivers.iloc[best_position]
    snap_distance_m = distances.iloc[best_pos_in_candidates]

    return nearest_reach["HYRIV_ID"], snap_distance_m


def trace_downstream(next_down_lookup, start_hyriv_id):
    """
    Follow NEXT_DOWN from start_hyriv_id until it terminates (NEXT_DOWN == 0)
    or the ID is not found. Returns the ordered list of HYRIV_ID from the dam
    down to (and including) the terminal reach, plus a status string.
    """
    chain = []
    visited = set()
    current = start_hyriv_id
    status = "OK"

    for _ in range(MAX_CHAIN_STEPS):
        if current not in next_down_lookup.index:
            status = "BROKEN_CHAIN"  # NEXT_DOWN pointed to an ID not in this layer
            break
        if current in visited:
            status = "CYCLE_DETECTED"
            break

        chain.append(current)
        visited.add(current)
        next_id = next_down_lookup.loc[current]

        if next_id == 0:
            status = "REACHED_OUTLET"
            break
        current = next_id
    else:
        status = "MAX_STEPS_EXCEEDED"

    return chain, status


def extract_downstream_course(dam_id, dam_name, lat, lon, rivers, rivers_metric,
                               next_down_lookup, radius_m):
    """Run the full snap -> trace -> extract pipeline for a single dam."""
    qc = {
        "Dam ID": dam_id,
        "Dam name": dam_name,
        "Latitude": lat,
        "Longitude": lon,
        "start_HYRIV_ID": None,
        "snap_distance_m": None,
        "n_reaches": 0,
        "total_length_km": 0.0,
        "outlet_HYRIV_ID": None,
        "outlet_endorheic": None,
        "trace_status": None,
        "QC_flag": "OK",
    }

    try:
        start_id, snap_m = find_nearest_reach_within_radius(
            rivers, rivers_metric, lat, lon, radius_m
        )

        if start_id is None:
            qc["QC_flag"] = "NO_REACH_WITHIN_RADIUS"
            qc["trace_status"] = "NO_REACH_WITHIN_RADIUS"
            return gpd.GeoDataFrame(), qc

        qc["start_HYRIV_ID"] = start_id
        qc["snap_distance_m"] = round(snap_m, 1)

        chain_ids, status = trace_downstream(next_down_lookup, start_id)
        qc["trace_status"] = status
        if status != "REACHED_OUTLET":
            qc["QC_flag"] = status

        downstream_gdf = rivers[rivers["HYRIV_ID"].isin(chain_ids)].copy()
        order_map = {hid: i for i, hid in enumerate(chain_ids)}
        downstream_gdf["FLOW_ORDER"] = downstream_gdf["HYRIV_ID"].map(order_map)
        downstream_gdf = downstream_gdf.sort_values("FLOW_ORDER").reset_index(drop=True)
        downstream_gdf.insert(0, "Dam_ID", dam_id)

        qc["n_reaches"] = len(downstream_gdf)
        qc["total_length_km"] = round(downstream_gdf["LENGTH_KM"].sum(), 2)
        if len(downstream_gdf):
            outlet_row = downstream_gdf.iloc[-1]
            qc["outlet_HYRIV_ID"] = outlet_row["HYRIV_ID"]
            qc["outlet_endorheic"] = bool(outlet_row["ENDORHEIC"])

        return downstream_gdf, qc

    except Exception as e:
        qc["QC_flag"] = "FAILED"
        qc["trace_status"] = str(e)
        return gpd.GeoDataFrame(), qc


# ---------------------------------------------------------------------------
# Batch driver
# ---------------------------------------------------------------------------

def run_batch(dams_csv_arg, rivers_path, radius_m, qc_name=None):
    dams_csv = resolve_input_path(dams_csv_arg, DATA_DIR)
    csv_stem = dams_csv.stem  # CSV filename without extension

    plot_outdir = PLOT_DIR / csv_stem
    plot_outdir.mkdir(parents=True, exist_ok=True)

    if not qc_name:
        qc_name = f"{csv_stem}_output.csv"
    qc_path = OUTPUT_DIR / qc_name
    qc_path.parent.mkdir(parents=True, exist_ok=True)

    dams = pd.read_csv(dams_csv)
    rivers, rivers_metric, next_down_lookup = load_river_network(rivers_path)

    qc_records = []
    for i, row in dams.iterrows():
        dam_id = row["Dam ID"]
        dam_name = row.get("Dam name", "")
        print(f"[{i + 1}/{len(dams)}] Dam ID {dam_id} ({dam_name})...")

        downstream_gdf, qc = extract_downstream_course(
            dam_id, dam_name, row["Latitude"], row["Longitude"],
            rivers, rivers_metric, next_down_lookup, radius_m,
        )
        qc_records.append(qc)

        if len(downstream_gdf):
            out_path = plot_outdir / f"{dam_id}_WaterCourse.gpkg"
            downstream_gdf.to_file(out_path, driver="GPKG")

        if (i + 1) % SAVE_EVERY == 0:
            pd.DataFrame(qc_records).to_csv(qc_path, index=False)

    pd.DataFrame(qc_records).to_csv(qc_path, index=False)
    print(f"Done. Per-dam GeoPackages in {plot_outdir}, QC written to {qc_path}")


def open_file_dialog(initial_dir):
    """
    Open a native 'choose file' window for selecting the dam CSV.
    Returns the chosen path, or None if unavailable/cancelled.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        return None

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)  # bring the dialog to the front

    chosen = filedialog.askopenfilename(
        title="Select dam CSV file",
        initialdir=str(initial_dir),
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )
    root.destroy()

    return chosen if chosen else None


def choose_dams_csv():
    """Open a native file-picker window for the dam CSV; fall back to typed
    input if a GUI isn't available (e.g. running over SSH with no display)."""
    chosen = open_file_dialog(DATA_DIR)

    if chosen:
        return chosen

    print("File picker unavailable or cancelled — enter the path manually.")
    manual = input(f"Dam CSV filename (looked up in {DATA_DIR}) or full path: ").strip()
    while not manual:
        manual = input("Please enter a CSV filename or path: ").strip()
    return manual


def run_batch_interactive():
    dams_csv = choose_dams_csv()

    rivers_path = input(f"River network path [{DEFAULT_RIVERS_PATH}]: ").strip()
    rivers_path = rivers_path if rivers_path else DEFAULT_RIVERS_PATH

    radius_str = input(f"Search radius in meters [{SEARCH_RADIUS_M_DEFAULT:.0f}]: ").strip()
    radius_m = float(radius_str) if radius_str else SEARCH_RADIUS_M_DEFAULT

    default_qc_name = f"{Path(dams_csv).stem}_output.csv"
    qc_name = input(f"QC CSV filename [{default_qc_name}]: ").strip()
    qc_name = qc_name if qc_name else default_qc_name

    run_batch(dams_csv, rivers_path, radius_m, qc_name)


def run_single():
    rivers_path = input(f"Path to river network [{DEFAULT_RIVERS_PATH}]: ").strip()
    rivers_path = rivers_path if rivers_path else DEFAULT_RIVERS_PATH
    dam_id = input("Dam ID: ").strip()
    lat = float(input("Dam latitude: ").strip())
    lon = float(input("Dam longitude: ").strip())
    radius_m = input(f"Search radius in meters [{SEARCH_RADIUS_M_DEFAULT:.0f}]: ").strip()
    radius_m = float(radius_m) if radius_m else SEARCH_RADIUS_M_DEFAULT

    rivers, rivers_metric, next_down_lookup = load_river_network(rivers_path)
    downstream_gdf, qc = extract_downstream_course(
        dam_id, "", lat, lon, rivers, rivers_metric, next_down_lookup, radius_m
    )

    print(qc)
    if len(downstream_gdf):
        out_path = PLOT_DIR / "single_dam" / f"{dam_id}_WaterCourse.gpkg"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        downstream_gdf.to_file(out_path, driver="GPKG")
        print(f"Saved {qc['n_reaches']} reaches ({qc['total_length_km']} km) to {out_path}")
    else:
        print("No downstream course extracted — see QC status above.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dams", help="Dam CSV — bare filename (resolved against Data/) or full path")
    parser.add_argument("--rivers", default=DEFAULT_RIVERS_PATH,
                         help=f"Path to HydroRIVERS shapefile or its containing folder (default: {DEFAULT_RIVERS_PATH})")
    parser.add_argument("--radius", type=float, default=SEARCH_RADIUS_M_DEFAULT,
                         help=f"Search radius in meters for snapping dams to reaches (default {SEARCH_RADIUS_M_DEFAULT:.0f})")
    parser.add_argument("--qc", default=None,
                         help="QC CSV filename, saved under Output/ (default: {csv_name}_output.csv)")
    parser.add_argument("--single", action="store_true", help="Run single-dam interactive mode")
    args = parser.parse_args()

    if args.single:
        run_single()
    elif args.dams:
        run_batch(args.dams, args.rivers, args.radius, args.qc)
    else:
        run_batch_interactive()


if __name__ == "__main__":
    main()
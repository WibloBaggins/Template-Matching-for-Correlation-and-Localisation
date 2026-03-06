"""
generate_all_data.py — Full data generation pipeline

Generates ALL derivative products from raw inputs:
  1. Edge-detected overhead image from the greyscale overhead image
  2. Heightmaps, steepness maps, and edge-detected variants from radar point cloud CSVs

Run this FIRST before run_ue5_test.py if you want to regenerate from scratch.
If the pre-computed data already exists in data/UE5_radar/heightmaps_formatted_data_out_loop_5/,
you can skip this script entirely.

Usage:
    python generate_all_data.py
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2


# ──────────────────────────────────────────────────────────────
#  OVERHEAD IMAGE PROCESSING
# ──────────────────────────────────────────────────────────────

def generate_overhead_edges(input_path, output_path, low_threshold=50, high_threshold=150):
    """
    Apply Canny edge detection to the greyscale overhead image.

    Governing equation — Canny edge detection:
        1. Gaussian blur to suppress noise
        2. Gradient magnitude: G = sqrt(Gx^2 + Gy^2)  via Sobel operators
        3. Non-maximum suppression to thin edges
        4. Hysteresis thresholding: keep edges with gradient > high_threshold,
           and those > low_threshold that are connected to strong edges
    """
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load overhead image: {input_path}")

    edges = cv2.Canny(img, low_threshold, high_threshold)
    cv2.imwrite(output_path, edges)
    print(f"[overhead] Edge-detected image saved: {output_path}")
    return edges


# ──────────────────────────────────────────────────────────────
#  RADAR POINT CLOUD → HEIGHTMAP PIPELINE
# ──────────────────────────────────────────────────────────────

def combine_csv_files_cumulatively(input_folder, time_increment):
    """
    Combines CSV files from the input folder cumulatively based on time increments.

    Each CSV filename encodes a timestamp: robot_output_data_HH_MM_SS_ms.csv
    Files are accumulated into growing windows of `time_increment` seconds.

    Yields:
        cumulative_data (DataFrame): All point cloud data up to this time window
        current_time_window (int): Window index (multiply by time_increment for seconds)
    """
    files = sorted(os.listdir(input_folder))
    cumulative_data = pd.DataFrame(columns=['x', 'y', 'z'])
    current_time_window = 0

    for filename in files:
        match = re.match(r'robot_output_data_(\d{1,2})_(\d{1,2})_(\d{1,2})_(\d{1,3})\.csv', filename)
        if match:
            hours, minutes, seconds, ms = map(int, match.groups())
            total_seconds = hours * 3600 + minutes * 60 + seconds + ms / 1000.0
            time_window = int(total_seconds // time_increment)

            if time_window > current_time_window:
                yield cumulative_data, current_time_window
                current_time_window = time_window

            filepath = os.path.join(input_folder, filename)
            df = pd.read_csv(filepath)
            cumulative_data = pd.concat([cumulative_data, df], ignore_index=True)

    if not cumulative_data.empty:
        yield cumulative_data, current_time_window


def create_heightmap(data):
    """
    Rasterise a point cloud into a 2D heightmap at 1 meter per pixel.

    For each grid cell, takes the z-value of the point falling in that cell.
    Cells with no data are NaN.

    Governing equation — rasterisation:
        grid_cell(x, y) = z   where (x, y) = round(point_x - min_x), round(point_y - min_y)
    """
    if data.empty:
        return None, None, None, None, None

    x = data['x'].values
    y = data['y'].values
    z = data['z'].values

    min_x, max_x = np.floor(np.min(x)), np.ceil(np.max(x))
    min_y, max_y = np.floor(np.min(y)), np.ceil(np.max(y))

    width = int(max_x - min_x) + 1
    height = int(max_y - min_y) + 1

    heightmap = np.full((height, width), np.nan)

    for i in range(len(x)):
        ix = int(np.round(x[i] - min_x))
        iy = int(np.round(y[i] - min_y))
        if 0 <= ix < width and 0 <= iy < height:
            heightmap[iy, ix] = z[i]

    return heightmap, min_x, min_y, max_x, max_y


def interpolate_heightmap(heightmap):
    """
    Fill NaN gaps in the heightmap using OpenCV inpainting (no scipy needed).

    Uses Navier-Stokes based inpainting (cv2.INPAINT_NS) which propagates
    values from known regions into unknown (NaN) regions by solving:
        ∇²(∇²I) = 0   (biharmonic smoothness)
    subject to boundary conditions from the known pixel values.

    This replaces the original scipy.interpolate.griddata approach with a
    pure OpenCV solution that requires no additional dependencies.
    """
    # Build a mask of NaN pixels (255 = needs inpainting, 0 = known)
    nan_mask = np.isnan(heightmap).astype(np.uint8) * 255

    # Normalise known values to 0-255 for inpainting
    valid = ~np.isnan(heightmap)
    if not np.any(valid):
        return np.zeros_like(heightmap)

    vmin = np.nanmin(heightmap)
    vmax = np.nanmax(heightmap)
    rng = vmax - vmin if vmax > vmin else 1.0

    # Scale to uint8 for cv2.inpaint
    normalised = np.zeros_like(heightmap)
    normalised[valid] = (heightmap[valid] - vmin) / rng
    img_u8 = (normalised * 255).astype(np.uint8)

    # Inpaint with radius proportional to gap size (cap at 20 for speed)
    inpainted_u8 = cv2.inpaint(img_u8, nan_mask, inpaintRadius=10, flags=cv2.INPAINT_NS)

    # Scale back to original value range
    result = inpainted_u8.astype(np.float64) / 255.0 * rng + vmin

    # Preserve original known values exactly (avoid quantisation drift)
    result[valid] = heightmap[valid]

    return result


def compute_steepness(heightmap):
    """
    Compute gradient magnitude (steepness) of the heightmap.

    Governing equation:
        steepness(x, y) = sqrt( (dz/dx)^2 + (dz/dy)^2 )

    where dz/dx and dz/dy are computed via numpy central differences.
    """
    grad_y, grad_x = np.gradient(heightmap)
    steepness = np.sqrt(grad_x**2 + grad_y**2)
    steepness = np.where(np.isnan(heightmap), np.nan, steepness)
    return steepness


def save_coloured_map(data_array, output_path, cmap='viridis'):
    """Save a 2D array as a coloured PNG using a matplotlib colourmap."""
    valid = ~np.isnan(data_array)
    normalised = np.zeros_like(data_array)
    if np.any(valid):
        vmin = np.nanmin(data_array[valid])
        vmax = np.nanmax(data_array[valid])
        if vmax > vmin:
            normalised[valid] = (data_array[valid] - vmin) / (vmax - vmin)
    normalised[~valid] = 127 / 255.0

    colormap = plt.get_cmap(cmap)
    colour_image = colormap(normalised)
    colour_image = (colour_image[:, :, :3] * 255).astype(np.uint8)

    Image.fromarray(colour_image).save(output_path)


def save_edge_images(source_path, output_prefix, main_folder, subfolder, low_thresh, high_thresh, mask):
    """
    Apply Canny edge detection and save both masked and unmasked versions.

    Masked: NaN areas set to 127 (grey) so template matcher treats them as neutral.
    Unmasked: raw edges, interpolation fills the gaps.
    """
    img = cv2.imread(source_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"  [warn] Could not load {source_path}")
        return

    edges = cv2.Canny(img, low_thresh, high_thresh)

    # Masked version: set NaN regions to 127
    masked_edges = np.where(mask, 127, edges).astype(np.uint8)
    # Unmasked version: raw edges everywhere
    unmasked_edges = edges

    for variant, edge_img in [("masked", masked_edges), ("unmasked", unmasked_edges)]:
        out_dir = os.path.join(main_folder, f"{subfolder}_edge_{variant}")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{output_prefix}_edge.png")
        cv2.imwrite(out_path, edge_img)


def save_corner_info(min_x, min_y, max_x, max_y, main_folder, time_label, mask):
    """
    Save bounding box corner coordinates and unmasked area count.

    The coordinate transform inverts Y so that image row 0 = top of the real-world area:
        image_height = max_y_grid - min_y_grid
        min_y_inverted = image_height - min_y_grid
        max_y_inverted = image_height - max_y_grid
    """
    corner_dir = os.path.join(main_folder, 'corner_info')
    os.makedirs(corner_dir, exist_ok=True)

    min_x_g = np.floor(min_x)
    min_y_g = np.floor(min_y)
    max_x_g = np.ceil(max_x)
    max_y_g = np.ceil(max_y)

    image_height = max_y_g - min_y_g
    min_y_inv = image_height - min_y_g
    max_y_inv = image_height - max_y_g

    unmasked_area = int(np.sum(~mask))

    path = os.path.join(corner_dir, f'corner_info_{time_label}.txt')
    with open(path, 'w') as f:
        f.write(f'Top-left: ({min_x_g}, {max_y_inv})\n')
        f.write(f'Top-right: ({max_x_g}, {max_y_inv})\n')
        f.write(f'Bottom-left: ({min_x_g}, {min_y_inv})\n')
        f.write(f'Bottom-right: ({max_x_g}, {min_y_inv})\n')
        f.write(f'Unmasked Area (in pixels): {unmasked_area}\n')


def process_one_timestep(data, time_label, main_folder,
                         edge_params_steepness=(50, 150),
                         edge_params_sos=(150, 150),
                         edge_params_heightmap=(30, 30)):
    """
    Full processing pipeline for a single cumulative time window.

    Produces: heightmap, steepness, steepness-of-steepness, and Canny edges
    of each (masked + unmasked), plus corner_info metadata.
    """
    heightmap, min_x, min_y, max_x, max_y = create_heightmap(data)
    if heightmap is None:
        print(f"  [skip] No data for {time_label}")
        return

    mask = np.isnan(heightmap)
    interpolated = interpolate_heightmap(heightmap)

    prefix = f'heightmap_{time_label}'

    # --- Heightmap ---
    hm_dir = os.path.join(main_folder, 'heightmap')
    os.makedirs(hm_dir, exist_ok=True)
    hm_path = os.path.join(hm_dir, f'{prefix}.png')
    save_coloured_map(interpolated, hm_path)

    # --- Steepness ---
    steepness = compute_steepness(interpolated)
    st_dir = os.path.join(main_folder, 'steepness')
    os.makedirs(st_dir, exist_ok=True)
    st_path = os.path.join(st_dir, f'{prefix}_steepness.png')
    save_coloured_map(steepness, st_path)

    # --- Steepness of steepness ---
    sos = compute_steepness(steepness)
    sos_dir = os.path.join(main_folder, 'steepness_of_steepness')
    os.makedirs(sos_dir, exist_ok=True)
    sos_path = os.path.join(sos_dir, f'{prefix}_steepness_of_steepness.png')
    save_coloured_map(sos, sos_path)

    # --- Edge detection on all three ---
    save_edge_images(st_path, f'{prefix}_steepness', main_folder, 'steepness', *edge_params_steepness, mask)
    save_edge_images(sos_path, f'{prefix}_steepness_of_steepness', main_folder, 'steepness_of_steepness', *edge_params_sos, mask)
    save_edge_images(hm_path, prefix, main_folder, 'heightmap', *edge_params_heightmap, mask)

    # --- Corner info ---
    save_corner_info(min_x, min_y, max_x, max_y, main_folder, time_label, mask)

    print(f"  [done] {time_label}: {interpolated.shape[1]}x{interpolated.shape[0]}px, "
          f"{int(np.sum(~mask))} known pixels")


# ──────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────

def main():
    base_dir = "data"

    # --- Step 1: Generate edge-detected overhead image ---
    greyscale_path = os.path.join(base_dir, "visual_band", "greyscale_1_pixel_per_square_meter.png")
    edges_path = os.path.join(base_dir, "visual_band", "edges_greyscale_1_pixel_per_square_meter.png")

    if not os.path.exists(greyscale_path):
        print(f"ERROR: Raw overhead image not found at {greyscale_path}")
        return

    print("=" * 60)
    print("STEP 1: Generate overhead edge-detected image")
    print("=" * 60)
    generate_overhead_edges(greyscale_path, edges_path, low_threshold=50, high_threshold=150)

    # --- Step 2: Generate heightmaps from radar point clouds ---
    input_folder = os.path.join(base_dir, "UE5_radar", "formatted_data_out_loop_5")
    output_folder = os.path.join(base_dir, "UE5_radar", "heightmaps_formatted_data_out_loop_5")

    if not os.path.exists(input_folder):
        print(f"ERROR: Formatted point cloud folder not found at {input_folder}")
        return

    print()
    print("=" * 60)
    print("STEP 2: Generate heightmaps from radar point clouds")
    print(f"  Input:  {input_folder}  ({len(os.listdir(input_folder))} CSV files)")
    print(f"  Output: {output_folder}")
    print("=" * 60)

    time_increment = 5  # seconds per cumulative window

    for cumulative_data, time_window in combine_csv_files_cumulatively(input_folder, time_increment):
        time_label = f'{int(time_window * time_increment)}_seconds'
        process_one_timestep(cumulative_data, time_label, output_folder)

    print()
    print("=" * 60)
    print("GENERATION COMPLETE")
    print(f"  Overhead edges: {edges_path}")
    print(f"  Heightmap products: {output_folder}")
    print("=" * 60)


if __name__ == "__main__":
    main()

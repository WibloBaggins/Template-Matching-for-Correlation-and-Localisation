"""
run_ue5_test.py — UE5 Simulated Radar Localisation Test (Chapter 5)

Matches edge-detected radar heightmaps against an overhead satellite edge image
using OpenCV TM_CCOEFF_NORMED template matching.

Outputs:
  1. Statistical scatter plot: distance error vs known area, with trendlines
  2. Full-map visualisation: matched location highlighted on the overhead image
  3. Zoomed overlay: radar edge map (red) overlaid on overhead edge map (white)

Usage:
    python run_ue5_test.py
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
def linregress(x, y):
    """Linear regression using numpy polyfit (no scipy dependency)."""
    coeffs = np.polyfit(x, y, 1)
    return coeffs[0], coeffs[1], None, None, None
import matplotlib.lines as mlines


# ──────────────────────────────────────────────────────────────
#  UTILITY FUNCTIONS
# ──────────────────────────────────────────────────────────────

def load_corner_info(corner_info_file):
    """Load corner info: bounding box coordinates + unmasked pixel count."""
    with open(corner_info_file, 'r') as f:
        corners = f.readlines()
    corner_info = {
        line.split(':')[0].lower().replace('-', '_'):
        tuple(map(float, line.split(': ')[1].strip('()\n').split(',')))
        for line in corners[:4]
    }
    corner_info['unmasked_area'] = int(corners[4].split(': ')[1])
    return corner_info


def calculate_distance_error(point1, point2):
    """
    Euclidean distance between two points.

    error = sqrt( (x1 - x2)^2 + (y1 - y2)^2 )
    """
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def crop_center(img, scale):
    """Crop the center of the image by the given scale factor."""
    cy, cx = img.shape[0] // 2, img.shape[1] // 2
    ry, rx = int(cy * scale), int(cx * scale)
    return img[cy - ry:cy + ry, cx - rx:cx + rx]


# ──────────────────────────────────────────────────────────────
#  VISUALISATION: RED OVERLAY + FULL MAP HIGHLIGHT
# ──────────────────────────────────────────────────────────────

def create_match_visualisation(target_img, template_img, max_loc, corner_info,
                               target_center, distance_error, method_name, time_label,
                               output_dir):
    """
    Create a 3-panel visualisation figure:
      Top-left:    Context map — full overhead with matched region boxed + zoom indicator
      Bottom-left: Zoomed overhead — the matched region from the satellite, greyscale
      Right:       Red/white overlay — radar edges (red) on top of overhead edges (white)

    Args:
        target_img:     greyscale overhead edge image (rotated)
        template_img:   greyscale radar edge image
        max_loc:        (x, y) top-left of best match in target image
        corner_info:    bounding box metadata dict
        target_center:  (x, y) center of target = true origin
        distance_error: distance in meters
        method_name:    preprocessing method name
        time_label:     e.g. "120_seconds"
        output_dir:     folder to save the figure
    """
    from matplotlib.patches import Patch, Rectangle, FancyArrowPatch
    import matplotlib.gridspec as gridspec

    th, tw = template_img.shape[:2]
    mx, my = max_loc

    # --- Compute matched origin in target image ---
    template_real_origin = (-corner_info['top_left'][0], corner_info['top_left'][1])
    matched_origin_px = (int(mx + template_real_origin[0]),
                         int(my + template_real_origin[1]))

    # --- Pad to add context around the matched region ---
    pad = max(th, tw)  # generous padding around the match
    crop_x1 = max(mx - pad, 0)
    crop_y1 = max(my - pad, 0)
    crop_x2 = min(mx + tw + pad, target_img.shape[1])
    crop_y2 = min(my + th + pad, target_img.shape[0])

    # === PANEL 1 (top-left): Context map ===
    context_map = cv2.cvtColor(target_img, cv2.COLOR_GRAY2BGR)

    # Green rectangle around matched region
    cv2.rectangle(context_map, (mx, my), (mx + tw, my + th), (0, 255, 0), 4)

    # Orange rectangle around the zoomed crop area
    cv2.rectangle(context_map, (crop_x1, crop_y1), (crop_x2, crop_y2), (0, 165, 255), 2)

    # Crosses: matched origin (red) and true origin (cyan)
    draw_cross_cv(context_map, matched_origin_px, color=(0, 0, 255), size=20, thickness=4)
    draw_cross_cv(context_map, target_center, color=(255, 255, 0), size=20, thickness=4)
    cv2.line(context_map, matched_origin_px, target_center, (0, 165, 255), 3, cv2.LINE_AA)

    # === PANEL 2 (bottom-left): Zoomed overhead crop ===
    zoomed_overhead = target_img[crop_y1:crop_y2, crop_x1:crop_x2]
    zoomed_bgr = cv2.cvtColor(zoomed_overhead, cv2.COLOR_GRAY2BGR)

    # Draw green rectangle around the match region within the crop
    local_mx = mx - crop_x1
    local_my = my - crop_y1
    cv2.rectangle(zoomed_bgr, (local_mx, local_my), (local_mx + tw, local_my + th),
                  (0, 255, 0), 2)

    # Matched origin cross (red) within crop
    local_origin = (matched_origin_px[0] - crop_x1, matched_origin_px[1] - crop_y1)
    draw_cross_cv(zoomed_bgr, local_origin, color=(0, 0, 255), size=12, thickness=2)

    # True origin cross (cyan) within crop — only if visible
    local_true = (target_center[0] - crop_x1, target_center[1] - crop_y1)
    if 0 <= local_true[0] < zoomed_bgr.shape[1] and 0 <= local_true[1] < zoomed_bgr.shape[0]:
        draw_cross_cv(zoomed_bgr, local_true, color=(255, 255, 0), size=12, thickness=2)

    # === PANEL 3 (right): Red/white overlay ===
    target_crop = target_img[my:my + th, mx:mx + tw]
    overlay = cv2.cvtColor(target_crop, cv2.COLOR_GRAY2BGR)

    # Radar edges become red
    radar_mask = template_img > 0
    overlay[radar_mask] = (0, 0, 255)  # BGR red

    # === COMPOSE FIGURE (2 rows left, 1 tall right) ===
    fig = plt.figure(figsize=(22, 12), facecolor='#1a1a1a')
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1.5], height_ratios=[1, 1],
                           hspace=0.15, wspace=0.08)

    # Top-left: context map
    ax_ctx = fig.add_subplot(gs[0, 0])
    ax_ctx.imshow(cv2.cvtColor(context_map, cv2.COLOR_BGR2RGB))
    ax_ctx.set_title("Full Overhead Map", fontsize=12, color='white', pad=8)
    ax_ctx.axis('off')

    # Legend
    legend_elements = [
        Patch(facecolor='lime', edgecolor='green', label='Matched region'),
        Patch(facecolor='red', label='Matched origin'),
        Patch(facecolor='cyan', label='True origin'),
        Patch(facecolor='orange', label=f'Error: {distance_error:.1f}m'),
    ]
    ax_ctx.legend(handles=legend_elements, loc='lower right', fontsize=8,
                  framealpha=0.85, facecolor='#2a2a2a', edgecolor='#555',
                  labelcolor='white')

    # Bottom-left: zoomed overhead
    ax_zoom = fig.add_subplot(gs[1, 0])
    ax_zoom.imshow(cv2.cvtColor(zoomed_bgr, cv2.COLOR_BGR2RGB))
    ax_zoom.set_title("Zoomed Match Region (Overhead)", fontsize=12,
                      color='white', pad=8)
    ax_zoom.axis('off')

    # Right (spanning both rows): red/white overlay
    ax_overlay = fig.add_subplot(gs[:, 1])
    ax_overlay.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    ax_overlay.set_title("Edge Alignment Overlay\n"
                         "Red = Radar edges  |  White = Overhead edges",
                         fontsize=13, color='white', pad=10)
    ax_overlay.axis('off')

    # Suptitle
    clean_method = method_name.replace('_', ' ').title()
    fig.suptitle(f"{clean_method}  —  {time_label.replace('_', ' ')}  —  "
                 f"Error: {distance_error:.1f}m  —  "
                 f"Known area: {corner_info['unmasked_area']} px",
                 fontsize=14, color='white', y=0.98, fontweight='bold')

    # Save
    os.makedirs(output_dir, exist_ok=True)
    safe_method = method_name.replace(os.sep, '_').replace('/', '_')
    filename = f"vis_{safe_method}_{time_label}.png"
    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)

    return save_path


def draw_cross_cv(img, center, color=(0, 0, 255), size=10, thickness=2):
    """Draw a cross on a BGR image at the given (x, y) center."""
    x, y = int(center[0]), int(center[1])
    cv2.line(img, (x - size, y), (x + size, y), color, thickness)
    cv2.line(img, (x, y - size), (x, y + size), color, thickness)


# ──────────────────────────────────────────────────────────────
#  CORE MATCHING
# ──────────────────────────────────────────────────────────────

def process_image_sets(image_sets, target, target_img_center, start_seconds,
                       highest_seconds, corner_info_dir,
                       cap_y_axis=False, max_y_value=10,
                       vis_output_dir=None):
    """
    Run template matching on all image sets, collect statistics,
    and optionally generate visualisations for the best matches.

    Template matching uses normalised cross-correlation:
        R(x,y) = sum[T'(x',y') * I'(x+x', y+y')] /
                 sqrt( sum[T'^2] * sum[I'^2] )
    where T' and I' are zero-mean template and image patches.
    """
    data_records = {}
    min_known_area_points = {}
    global_min_known_area = float('inf')
    global_min_point = None

    # Track the best match per method for visualisation
    best_matches = {}

    for subdir, images in image_sets:
        method_name = os.path.relpath(subdir, os.path.dirname(corner_info_dir))
        folder_name = os.path.basename(os.path.dirname(subdir))

        if method_name not in data_records:
            data_records[method_name] = []

        subfolder_data = []

        for img_file in images:
            if "s" not in img_file:
                continue

            img_path = os.path.join(subdir, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            current_seconds = int(img_file.split('_')[1])
            normalized_time = current_seconds - start_seconds

            corner_info_file = os.path.join(corner_info_dir,
                                            f"corner_info_{current_seconds}_seconds.txt")
            if not os.path.exists(corner_info_file):
                continue

            corner_info = load_corner_info(corner_info_file)
            unmasked_area = corner_info['unmasked_area']

            # --- Template matching ---
            result = cv2.matchTemplate(target, img, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            # --- Coordinate transform ---
            # template_real_origin: where (0,0) in real-world maps to in image space
            template_real_origin = (-corner_info['top_left'][0],
                                    corner_info['top_left'][1])
            matched_real_origin = (max_loc[0] + template_real_origin[0],
                                   max_loc[1] + template_real_origin[1])

            distance_error = calculate_distance_error(matched_real_origin,
                                                      target_img_center)

            subfolder_data.append((unmasked_area, distance_error, folder_name))
            data_records[method_name].append((normalized_time, distance_error,
                                              unmasked_area))

            # Track the best (lowest error) match with most area for vis
            time_label = f"{current_seconds}_seconds"
            key = method_name
            if key not in best_matches or distance_error < best_matches[key]['error']:
                best_matches[key] = {
                    'error': distance_error,
                    'max_loc': max_loc,
                    'template_img': img,
                    'corner_info': corner_info,
                    'time_label': time_label,
                    'method_name': method_name,
                    'area': unmasked_area,
                }

        if subfolder_data:
            subfolder_data.sort(key=lambda x: -x[0])
            min_point = None
            for i in range(len(subfolder_data)):
                area, error, fn = subfolder_data[i]
                if error <= max_y_value:
                    if i == len(subfolder_data) - 1 or subfolder_data[i + 1][1] > max_y_value:
                        min_point = (area, error, fn)
                        break

            if min_point:
                if min_point[0] < global_min_known_area:
                    global_min_known_area = min_point[0]
                    global_min_point = min_point

                if method_name not in min_known_area_points:
                    min_known_area_points[method_name] = []
                min_known_area_points[method_name].append(min_point)

    # Mark global min
    for method_name, points in min_known_area_points.items():
        for i, (area, error, fn) in enumerate(points):
            is_global = (area == global_min_known_area)
            points[i] = (area, error, fn, is_global)

    # --- Generate visualisations for best matches ---
    if vis_output_dir:
        print(f"\n  Generating match visualisations...")
        for key, info in best_matches.items():
            save_path = create_match_visualisation(
                target_img=target,
                template_img=info['template_img'],
                max_loc=info['max_loc'],
                corner_info=info['corner_info'],
                target_center=target_img_center,
                distance_error=info['error'],
                method_name=info['method_name'],
                time_label=info['time_label'],
                output_dir=vis_output_dir
            )
            print(f"    Saved: {save_path}  (error: {info['error']:.1f}m, "
                  f"area: {info['area']}px)")

    return data_records, min_known_area_points


# ──────────────────────────────────────────────────────────────
#  PLOTTING
# ──────────────────────────────────────────────────────────────

def plot_results(aggregated_data, min_known_area_points, target_area,
                 show_labels=False, cap_y_axis=False, max_y_value=10,
                 output_dir=None):
    """
    Plot ALL data points: error vs known scanned area with trendlines per method.
    Optionally marks the smallest-area point that achieves <max_y_value error.
    """

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111)
    ax.set_title("Template Matching: Distance Error vs Known Scanned Area",
                 fontsize=14, fontweight='bold')

    color_map = plt.get_cmap('tab10')
    colors = {}

    # Plot ALL data points from aggregated_data
    for idx, (method, records) in enumerate(sorted(aggregated_data.items())):
        if not records:
            continue

        # records: list of (normalized_time, distance_error, unmasked_area)
        areas = [r[2] for r in records]
        errors = [r[1] for r in records]

        color = color_map(idx)
        colors[method] = color

        ax.scatter(areas, errors, color=color, marker='o', s=40, alpha=0.6,
                   zorder=3, label=method)

        # Trendline
        if len(areas) > 1:
            slope, intercept, _, _, _ = linregress(
                np.array(areas, dtype=float), np.array(errors, dtype=float))
            x_fit = np.linspace(min(areas), max(areas), 100)
            ax.plot(x_fit, x_fit * slope + intercept, '-', color=color,
                    alpha=0.8, linewidth=2, zorder=4)

    # Mark threshold transition points if any exist
    for method, points in min_known_area_points.items():
        if not points:
            continue
        color = colors.get(method, 'black')
        for area, error, fn, is_global in points:
            marker = 'X' if is_global else 'D'
            ax.scatter(area, error, color=color, marker=marker, s=200,
                       edgecolors='black', linewidths=1.5, zorder=6)

    ax.set_xlabel('Known Scanned Area (m²)', fontsize=12)
    ax.set_ylabel('Localisation Error (meters)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    if cap_y_axis:
        ax.set_ylim(bottom=0, top=max_y_value)
    else:
        ax.set_ylim(bottom=0)

    # Secondary x-axis: percentage of target area
    ax2 = ax.secondary_xaxis('top')
    ax.callbacks.connect('xlim_changed', lambda a: _update_pct_axis(a, ax2, target_area))
    _update_pct_axis(ax, ax2, target_area)
    ax2.set_xlabel('Percentage of Target Area', fontsize=11)

    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "results_error_vs_area.png")
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n  Results plot saved: {save_path}")

    plt.close(fig)


def _update_pct_axis(ax, ax2, target_area):
    """Keep secondary percentage axis in sync with primary."""
    lo, hi = ax.get_xlim()
    ax2.set_xlim(lo, hi)
    ticks = ax.get_xticks()
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([f"{100 * t / target_area:.1f}%" for t in ticks])


# ──────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────

def main():
    base_dir = "data"
    output_dir = "output"
    vis_dir = os.path.join(output_dir, "match_visualisations")

    target_image_path = os.path.join(base_dir, "visual_band",
                                     "edges_greyscale_1_pixel_per_square_meter.png")
    target_area = 2000 * 2000  # total target area in pixels (= m^2 at 1m/px)

    # Filter: only use unmasked edge variants
    must_contain = ["unmasked"]
    must_not_contain = ["frogs"]

    max_y_value = 10

    # --- Load target ---
    target_img = cv2.imread(target_image_path, cv2.IMREAD_GRAYSCALE)
    if target_img is None:
        print(f"ERROR: Target image not found: {target_image_path}")
        print("  Run generate_all_data.py first to create it from the raw overhead image.")
        return

    target_img = cv2.rotate(target_img, cv2.ROTATE_90_CLOCKWISE)
    print(f"Target image loaded: {target_img.shape[1]}x{target_img.shape[0]}px")

    # --- Find radar datasets ---
    radar_base = os.path.join(base_dir, "UE5_radar")
    radar_dirs = [os.path.join(radar_base, d)
                  for d in sorted(os.listdir(radar_base))
                  if d.startswith("heightmaps_")]

    if not radar_dirs:
        print(f"ERROR: No heightmap directories found in {radar_base}")
        print("  Run generate_all_data.py first.")
        return

    print(f"Radar directories: {[os.path.basename(d) for d in radar_dirs]}")

    aggregated_data = {}
    all_min_points = {}

    for ue5_dir in radar_dirs:
        corner_info_dir = os.path.join(ue5_dir, "corner_info")
        if not os.path.exists(corner_info_dir):
            print(f"  [skip] No corner_info in {ue5_dir}")
            continue

        image_sets = []
        highest_seconds = 0

        for subdir, _, files in os.walk(ue5_dir):
            if "corner_info" in subdir:
                continue
            if not any(m in subdir for m in must_contain):
                continue
            if any(m in subdir for m in must_not_contain):
                continue

            images = sorted([f for f in files if f.endswith(".png") and "edge" in f])
            if images:
                image_sets.append((subdir, images))
                for img_file in images:
                    sec = int(img_file.split('_')[1])
                    highest_seconds = max(highest_seconds, sec)

        if not image_sets:
            continue

        image_sets.sort(key=lambda x: int(x[1][0].split('_')[1]))
        start_seconds = int(image_sets[0][1][0].split('_')[1])

        target_center = (target_img.shape[1] // 2, target_img.shape[0] // 2)

        print(f"\n  Processing: {os.path.basename(ue5_dir)}")
        print(f"    Methods: {[os.path.relpath(s, ue5_dir) for s, _ in image_sets]}")
        print(f"    Time range: {start_seconds}s to {highest_seconds}s")

        data_records, min_points = process_image_sets(
            image_sets, target_img, target_center, start_seconds, highest_seconds,
            corner_info_dir, cap_y_axis=True, max_y_value=max_y_value,
            vis_output_dir=vis_dir
        )

        for method, records in data_records.items():
            aggregated_data.setdefault(method, []).extend(records)
        for method, points in min_points.items():
            all_min_points.setdefault(method, []).extend(points)

    if not aggregated_data:
        print("\nNo data processed. Check that heightmap directories exist.")
        return

    print(f"\n  Methods analysed: {list(aggregated_data.keys())}")
    print(f"  Total match evaluations: {sum(len(v) for v in aggregated_data.values())}")

    plot_results(aggregated_data, all_min_points, target_area,
                 show_labels=False, cap_y_axis=False, max_y_value=max_y_value,
                 output_dir=output_dir)

    print(f"\nAll outputs saved to: {output_dir}/")


if __name__ == "__main__":
    main()

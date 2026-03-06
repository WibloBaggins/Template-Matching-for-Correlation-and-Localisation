import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from time import time

# Define the image corners for the templates
TEMPLATE_INFO = {
    "narrow": {
        "bottom_left": (-1.0, -100.0),
        "size": (200.0 - (-1.0), 101.0 - (-100.0))
    },
    "wide": {
        "bottom_left": (-1.0, -500.0),
        "size": (1000.0 - (-1.0), 501.0 - (-500.0))
    }
}

def calculate_real_origin_offset(template_info):
    """
    Calculate the real origin offset from the top-left corner of the template.
    
    Arguments:
    template_info -- dictionary containing bottom_left and size of the template
    
    Returns:
    real_origin_offset -- tuple of offsets in x and y directions
    """
    real_origin_x = -template_info["bottom_left"][0]
    real_origin_y = -template_info["bottom_left"][1]
    return real_origin_x, real_origin_y

# Calculate real origin offsets for narrow and wide templates
narrow_real_origin_offset = calculate_real_origin_offset(TEMPLATE_INFO["narrow"])
wide_real_origin_offset = calculate_real_origin_offset(TEMPLATE_INFO["wide"])

def template_match(target_img, template_img):
    """
    Perform template matching using the cv2.TM_CCOEFF_NORMED method.
    
    Arguments:
    target_img -- the target image array
    template_img -- the template image array
    
    Returns:
    matches -- list of (best match location, match value)
    """
    result = cv2.matchTemplate(target_img, template_img, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    return [(max_loc, max_val)], template_img.shape[::-1]

def draw_cross(img, center, color_x=(255, 0, 0), color_y=(0, 255, 0), size=10, diagonal=False, thickness=4):
    """
    Draw a cross on the image at the specified center.
    
    Arguments:
    img -- the image array
    center -- the center of the cross
    color_x -- color of the cross on the x-axis
    color_y -- color of the cross on the y-axis
    size -- size of the cross
    diagonal -- whether to draw the cross diagonally (for templates)
    thickness -- thickness of the cross lines
    
    Returns:
    img_with_cross -- the image with the cross drawn
    """
    img_with_cross = img.copy()
    x_center, y_center = center

    # Draw the cross lines
    if diagonal:
        img_with_cross = cv2.line(img_with_cross, (x_center - size, y_center - size), (x_center + size, y_center + size), color_x, thickness)
        img_with_cross = cv2.line(img_with_cross, (x_center - size, y_center + size), (x_center + size, y_center - size), color_y, thickness)
    else:
        img_with_cross = cv2.line(img_with_cross, (x_center - size, y_center), (x_center + size, y_center), color_x, thickness)
        img_with_cross = cv2.line(img_with_cross, (x_center, y_center - size), (x_center, y_center + size), color_y, thickness)
    
    return img_with_cross

def calculate_real_distance(template_type, match_location, target_center):
    """
    Calculate the real-world distance between the template's real origin and the target's real origin.
    
    Arguments:
    template_type -- type of the template ('narrow' or 'wide')
    match_location -- the top-left corner of the best match location
    target_center -- the center of the target image
    
    Returns:
    real_distance -- the real-world distance
    matched_part_real_coords -- real-world coordinates of the matched part
    """
    if template_type == "narrow":
        real_origin_offset = narrow_real_origin_offset
    elif template_type == "wide":
        real_origin_offset = wide_real_origin_offset
    else:
        return 0, (0, 0)

    # Calculate the matched part's real coordinates in the target frame
    matched_part_real_coords = (
        match_location[0] + real_origin_offset[0],
        match_location[1] + real_origin_offset[1]
    )
    
    # Calculate the distance to the target's real origin (target_center)
    real_distance = np.linalg.norm(np.array(matched_part_real_coords) - np.array(target_center))
    
    return real_distance, matched_part_real_coords

def draw_rectangle_on_image(img, match_location, template_size, template_name, target_center, draw_crosses=True):
    """
    Draw a rectangle on the image at the best match location and draw crosses at the real (0, 0).
    
    Arguments:
    img -- the image array
    match_location -- the top-left corner of the best match location
    template_size -- the size of the template image
    template_name -- the name of the template image
    target_center -- the center of the target image
    draw_crosses -- whether to draw crosses at the real (0, 0) location
    
    Returns:
    img_with_rectangle -- the image with the rectangle and crosses drawn
    real_location -- the real-world coordinates of the match location
    distance -- the distance in meters
    """
    top_left = match_location
    bottom_right = (top_left[0] + template_size[0], top_left[1] + template_size[1])
    
    # Determine the type of template
    if "wide" in template_name:
        template_type = "wide"
    elif "narrow" in template_name:
        template_type = "narrow"
    else:
        template_type = None

    if template_type:
        # Calculate the real-world distance and coordinates
        real_distance, matched_part_real_coords = calculate_real_distance(template_type, match_location, target_center)
        pixel_distance = np.linalg.norm(np.array(match_location))
    else:
        matched_part_real_coords = (0, 0)
        real_distance = 0
        pixel_distance = 0

    img_with_rectangle = img.copy()
    # Draw rectangle for the match
    img_with_rectangle = cv2.rectangle(img_with_rectangle, top_left, bottom_right, (0, 0, 255), 2)  # Default color: Red
    
    if draw_crosses:
        # Draw cross at the top-left corner of the template match
        img_with_rectangle = draw_cross(img_with_rectangle, top_left, color_x=(255, 0, 0), color_y=(255, 0, 0), size=20, diagonal=True, thickness=8)
        
        # Draw cross at the real origin location of the template match
        real_origin_in_target = (
            int(match_location[0] + narrow_real_origin_offset[0]) if template_type == "narrow" else int(match_location[0] + wide_real_origin_offset[0]),
            int(match_location[1] + narrow_real_origin_offset[1]) if template_type == "narrow" else int(match_location[1] + wide_real_origin_offset[1])
        )
        img_with_rectangle = draw_cross(img_with_rectangle, real_origin_in_target, size=50, diagonal=True, thickness=8)

        # Draw cross at (0, 0) in real-world coordinates for the target image (vertical)
        img_with_rectangle = draw_cross(img_with_rectangle, target_center, color_x=(255, 165, 0), color_y=(255, 192, 203), size=100, diagonal=False, thickness=8)
    
    return img_with_rectangle, matched_part_real_coords, real_distance, pixel_distance

def mask_out_percentage(template_img, percentage, method='random'):
    """
    Mask out a given percentage of the template image by setting those pixels to 127.
    
    Arguments:
    template_img -- the template image array
    percentage -- the percentage of the image to mask out
    method -- the method of masking ('random', 'boundary', 'central', 'block')
    
    Returns:
    masked_img -- the masked template image array
    """
    masked_img = template_img.copy()
    height, width = template_img.shape
    num_pixels = height * width
    num_mask = int(num_pixels * percentage / 100)

    if num_mask == 0:
        return masked_img

    if method == 'random':
        # Get random indices to mask out
        mask_indices = np.random.choice(num_pixels, num_mask, replace=False)
        masked_img.flat[mask_indices] = 127

    elif method == 'boundary':
        # Calculate the size of the central unmasked square
        unmasked_percentage = 100 - percentage
        central_size_x = int(width * np.sqrt(unmasked_percentage / 100))
        central_size_y = int(height * np.sqrt(unmasked_percentage / 100))

        # Ensure central size is within the image dimensions
        central_size_x = max(1, min(central_size_x, width))
        central_size_y = max(1, min(central_size_y, height))

        # Calculate the top-left and bottom-right corners of the central square
        top_left_x = (width - central_size_x) // 2
        top_left_y = (height - central_size_y) // 2
        bottom_right_x = top_left_x + central_size_x
        bottom_right_y = top_left_y + central_size_y

        # Mask out everything outside the central square
        masked_img[:top_left_y, :] = 127  # Top boundary
        masked_img[bottom_right_y:, :] = 127  # Bottom boundary
        masked_img[:, :top_left_x] = 127  # Left boundary
        masked_img[:, bottom_right_x:] = 127  # Right boundary

    elif method == 'central':
        # Mask out the central region of the image
        center_x, center_y = width // 2, height // 2
        mask_size_x, mask_size_y = int(center_x * np.sqrt(percentage / 100)), int(center_y * np.sqrt(percentage / 100))
        masked_img[center_y-mask_size_y:center_y+mask_size_y, center_x-mask_size_x:center_x+mask_size_x] = 127

    elif method == 'block':
        # Divide the template into 3x3 grid
        grid_size_x = width // 3
        grid_size_y = height // 3
        block_size_x = int(grid_size_x * (percentage / 100))
        block_size_y = int(grid_size_y * (percentage / 100))

        # Ensure block size does not exceed grid size
        block_size_x = min(block_size_x, grid_size_x)
        block_size_y = min(block_size_y, grid_size_y)

        for i in range(3):
            for j in range(3):
                if grid_size_x - block_size_x > 0 and grid_size_y - block_size_y > 0:
                    start_x = i * grid_size_x + np.random.randint(0, grid_size_x - block_size_x)
                    start_y = j * grid_size_y + np.random.randint(0, grid_size_y - block_size_y)
                    masked_img[start_y:start_y + block_size_y, start_x:start_x + block_size_x] = 127
                else:
                    # If the block size is equal to or larger than the grid size, place it at the start
                    masked_img[j * grid_size_y:(j + 1) * grid_size_y, i * grid_size_x:(i + 1) * grid_size_x] = 127

    return masked_img



def main():
    # Configurable parameters
    y_axis_cap = 10  # Cap for y-axis in meters
    threshold_80_percent = 10  # Threshold in meters for 80% masked
    num_matches_per_point = 10  # Number of template matches per point with different random masks
    percentage_interval = 1  # Interval at which masking percentage is increased (e.g., 10% per step)

    # Start timer
    start_time = time()

    base_dir = "data"
    target_dir = os.path.join(base_dir, "visual_band")
    templates_dir = os.path.join(base_dir, "3d_scan")
    example_dir = os.path.join(base_dir, "masking_examples")
    os.makedirs(example_dir, exist_ok=True)
    
    visual_band_image = "edges_greyscale_1_pixel_per_square_meter.png"
    target_image_path = os.path.join(target_dir, visual_band_image)
    
    template_files = []
    for subdir, _, files in os.walk(templates_dir):
        for file in files:
            if file.endswith(".png"):
                template_files.append(os.path.join(subdir, file))

    target_img = cv2.imread(target_image_path, cv2.IMREAD_GRAYSCALE)
    target_img = cv2.rotate(target_img, cv2.ROTATE_90_CLOCKWISE)  # Rotate the image 90 degrees clockwise
    
    real_error_img = target_img.copy()
    target_center = (real_error_img.shape[1] // 2, real_error_img.shape[0] // 2)  # Center of the target image
    real_error_details = []
    match_count = 0

    results_mean = {
        'random': {},
        'boundary': {},
        'central': {},
        'block': {}
    }
    example_masked_templates_75 = {
        'random': None,
        'boundary': None,
        'central': None,
        'block': None
    }

    # Masking methods
    masking_methods = ['random', 'boundary', 'central', 'block']

    # Generate and save example images for 75% masking before template matching
    for method in masking_methods:
        for template_image_path in template_files:
            template_img = cv2.imread(template_image_path, cv2.IMREAD_GRAYSCALE)
            short_name = template_image_path.split("july")[-1]

            print(f"Creating 75% {method} example of template {short_name}")
            
            if template_img is None:
                print(f"Error: Failed to load template image from {template_image_path}")
                continue
            
            example_masked_templates_75[method] = mask_out_percentage(template_img, 75, method)
            
            if example_masked_templates_75[method] is None or example_masked_templates_75[method].size == 0:
                print(f"Error: Masked image for {method} was not created correctly.")
                continue
            
            example_image_path = os.path.join(example_dir, f"{method}_masked_75_{short_name}")
            
            cv2.imwrite(example_image_path, example_masked_templates_75[method])
            
            if not os.path.exists(example_image_path):
                print(f"Error: Failed to save masked template image to {example_image_path}")
            else:
                print(f"Saved to {example_image_path}")

    # Iterate through template files
    for j, template_image_path in enumerate(template_files):
        template_img = cv2.imread(template_image_path, cv2.IMREAD_GRAYSCALE)
        short_name = template_image_path.split("july")[-1]

        for method in masking_methods:
            print(f"Starting {method} masking")

            percentage_masked = 0

            real_distances_all = []
            percentages = []
            continue_to_100 = False

            while percentage_masked <= 100:
                distances_per_mask = []
                for _ in range(num_matches_per_point):  # Perform template matches with different random masks
                    masked_template_img = mask_out_percentage(template_img, percentage_masked, method)
                    matches, template_size = template_match(target_img, masked_template_img)
                    
                    for match_location, match_score in matches:
                        _, _, real_distance, _ = draw_rectangle_on_image(
                            real_error_img, match_location, template_size, template_image_path, target_center, draw_crosses=False
                        )
                        distances_per_mask.append(real_distance)

                mean_distance = np.mean(distances_per_mask)
                real_distances_all.append(mean_distance)
                percentages.append(percentage_masked)

                if percentage_masked == 90 and mean_distance <= 10:
                    continue_to_100 = True

                if percentage_masked >= 100:
                    break

                if percentage_masked % 10 == 0 or continue_to_100:
                    print(f"Template [{j+1}/{len(template_files)}], [{percentage_masked}%] percent masked, {short_name}, real distance: {mean_distance:.2f}")

                # Increment percentage by the specified interval or by 1% when approaching 100% masking
                percentage_masked += percentage_interval if not continue_to_100 else 1

            if real_distances_all:
                results_mean[method][template_image_path] = (percentages, real_distances_all)
        
        match_count += 1
        print(f"{match_count}/{len(template_files)} matches complete for {visual_band_image}")

    # Plot the results
    fig, axes = plt.subplots(4, 2, figsize=(24, 32))
    row_titles = ['Random Masking', 'Boundary Masking', 'Central Masking', 'Block Masking']

    for i, method in enumerate(masking_methods):
        # Plotting the masked template examples in the first column
        example_template_img_75 = example_masked_templates_75[method]
        if example_template_img_75 is not None:
            axes[i, 0].imshow(example_template_img_75, cmap='gray')
            axes[i, 0].set_title(f'75% Masked Template Example\n{method}')
            axes[i, 0].axis('off')
        else:
            print(f"Warning: No image found for {method} to display in the first column.")

        # Prepare to plot results in the second column
        filtered_results = []
        additional_results = []

        # Filter results that meet the threshold (under 10 meters at 80% masking)
        for template_path, (percentages, real_distances_mean) in results_mean[method].items():
            if real_distances_mean[percentages.index(80)] <= threshold_80_percent:
                filtered_results.append((template_path, percentages, real_distances_mean))
            else:
                additional_results.append((template_path, percentages, real_distances_mean))

        # If less than 5 results are filtered, add more from the additional results
        if len(filtered_results) < 5:
            additional_results.sort(
                key=lambda x: sum(1 for dist in x[2] if dist <= 10),
                reverse=True
            )
            filtered_results.extend(additional_results[:5 - len(filtered_results)])

        # Plot the filtered (and potentially expanded) results
        for template_path, percentages, real_distances_mean in filtered_results:
            axes[i, 1].plot(percentages, real_distances_mean, label=os.path.basename(template_path))

        axes[i, 1].set_xlabel('Percentage Masked Out')
        axes[i, 1].set_ylabel('Real Distance (meters)')
        axes[i, 1].set_ylim(0, y_axis_cap)
        axes[i, 1].set_title(f'Mean Template Matching Real Distance vs. Percentage Masked Out\n{row_titles[i]}')
        axes[i, 1].legend()
        axes[i, 1].grid(True)

    plt.tight_layout()
    plt.show()

    # End timer
    end_time = time()
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"Time taken: {int(minutes)} minutes and {int(seconds)} seconds")

if __name__ == "__main__":
    main()

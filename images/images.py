import os
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict

import numpy as np

ROOT_DIR = r"C:\Users\deniz\OneDrive\Desktop\images"

models = ["Linear Head", "Adapter", "SAM"]

columns = ["Boundary Map", "Instance Color Map", "Overlay"]

# Utility to collect images per sample
def collect_images():
    data = defaultdict(dict)

    for model in models:
        model_path = os.path.join(ROOT_DIR, model)
        if not os.path.exists(model_path):
            continue

        for city in os.listdir(model_path):
            city_path = os.path.join(model_path, city)

            if not os.path.isdir(city_path):
                continue
            for fname in os.listdir(city_path):
                if not fname.endswith((".png", ".jpg")):
                    continue
                base_id = "_".join(fname.split("_")[:3])  # e.g., lindau_000004_000019
                sample = data[model].setdefault(base_id, {})
                fpath = os.path.join(city_path, fname)

                if "boundary" in fname:
                    sample["boundary"] = fpath
                elif "instanceIds_color" in fname:
                    sample["instance"] = fpath
                elif "leftImg8bit" in fname:
                    sample["rgb"] = fpath

    return data

# Blend PIL images using alpha
def generate_overlay(rgb_path, instance_path):
    try:
        rgb = Image.open(rgb_path).convert("RGBA")
        label = Image.open(instance_path).convert("RGBA")
        
        # Resize both to match
        rgb = rgb.resize(label.size)
        
        # Apply individual opacities (adjust alpha channel)
        rgb.putalpha(int(0.65 * 255))
        label.putalpha(int(0.35 * 255))

        # First: overlay label onto rgb
        base = Image.alpha_composite(rgb, label)

        return base.convert("RGB")  # Convert back to RGB for matplotlib
    except Exception:
        return None

def get_all_sample_ids(image_data):
    # Collect all unique sample IDs across models
    sample_ids = set()
    for samples in image_data.values():
        sample_ids.update(samples.keys())
    return sample_ids

# Plotting function
def plot_table(image_data, sample_id, output_path):
    n_rows = len(image_data)
    n_cols = len(columns)

    # Slightly smaller height
    fig, axs = plt.subplots(
        n_rows + 1, n_cols + 1,
        figsize=(3.5 * (n_cols + 1), 1.8 * (n_rows + 1)),
        gridspec_kw={'height_ratios': [0.05] + [1] * n_rows,
                       'width_ratios': [0.4] + [1] * n_cols}
    )
    axs = np.atleast_2d(axs)

    # Turn off all axes on top row and first column
    for ax in axs[0]:
        ax.axis('off')
    for i in range(1, n_rows + 1):
        axs[i][0].axis('off')

    # Set column headers
    for j, col in enumerate(columns):
        axs[0][j + 1].set_title(col, fontsize=16, pad=0)  # pad controls space below header

    # Fill in rows
    for i, (model, samples) in enumerate(image_data.items(), 1):
        # Put model name in first column
        axs[i][0].text(0.5, 0.5, model, ha='center', va='center', fontsize=16)
        axs[i][0].axis('off')
        axs[i][0].set_aspect('equal')

        # Center text vertically by setting ylim
        axs[i][0].set_ylim(0, 1)

        # Get images for the row
        sample = samples.get(sample_id, {})
        imgs = []
        for key in ['boundary', 'instance']:
            path = sample.get(key)
            img = Image.open(path).convert("RGB") if path else None
            imgs.append(img)

        # Overlay image
        if sample.get("rgb") and sample.get("instance"):
            overlay = generate_overlay(sample["rgb"], sample["instance"])
        else:
            overlay = None
        imgs.append(overlay)

        # Plot each image cell
        for j in range(n_cols):
            axs[i][j + 1].axis('off')
            axs[i][j + 1].set_aspect('equal')
            axs[i][j + 1].margins(0)
            if imgs[j] is not None:
                axs[i][j + 1].imshow(imgs[j])

    # Adjust spacing - less vertical space
    fig.subplots_adjust(top=0.9, hspace=0, wspace=0.09, left=0.05, right=0.95)

    # Remove tight_layout to avoid conflict
    # plt.tight_layout(pad=1.0, h_pad=0.2, w_pad=0.4)

    plt.savefig(output_path, dpi=300)
    plt.close()



# Run
image_data = collect_images()
all_sample_ids = get_all_sample_ids(image_data)
for sample_id in list(all_sample_ids)[:5]:
    out_file = f"visual_table_{sample_id}.png"
    plot_table(image_data, sample_id, out_file)
    print(f"Saved visualization for sample {sample_id} as {out_file}")

# Load all images
image_paths = ["visual_table_frankfurt_000000_013942.png", 
               "visual_table_frankfurt_000000_016286.png", 
               "visual_table_lindau_000004_000019.png", 
               "visual_table_lindau_000024_000019.png", 
               "visual_table_munster_000024_000019.png", 
               "visual_table_munster_000078_000019.png"]
images = [Image.open(path) for path in image_paths]

# Define how many pixels to remove from the left
trim_left = 700
trim_top = 260
trim_right = 160
trim_bottom = 200

def trim_box(img_width, img_height):
    return (
        trim_left,
        trim_top,
        img_width - trim_right,
        img_height - trim_bottom
    )

trimmed_images = [
    img.crop(trim_box(*img.size)) for img in images
]

processed_pairs = [
    (trimmed_images[0], trimmed_images[1]),
    (trimmed_images[2], trimmed_images[3]),
    (trimmed_images[4], trimmed_images[5])
]

# Get new dimensions after trimming
new_width, new_height = processed_pairs[0][0].size

# Create a canvas for 3 rows of side-by-side images
canvas_width = new_width * 2
canvas_height = new_height * 3
grid_image = Image.new('RGB', (canvas_width, canvas_height))

# Paste image pairs onto canvas
for row, (img_left, img_right) in enumerate(processed_pairs):
    grid_image.paste(img_left, (0, row * new_height))
    grid_image.paste(img_right, (new_width, row * new_height))

# Save result
grid_image.save("comparison_grid_trimmed.jpg")
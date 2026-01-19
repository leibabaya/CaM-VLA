import pandas as pd
import os
from PIL import Image
from tqdm import tqdm
from pathlib import Path


def crop_and_pad_image(image_path, center_x, center_y, output_size=(256, 256)):
    """
    Crops an image around a center point with adaptive shifting at the boundaries.
    If the crop box exceeds the image dimensions, it is shifted to fit.

    Args:
        image_path (str or Path): The path to the source image.
        center_x (int): The x-coordinate of the desired crop center.
        center_y (int): The y-coordinate of the desired crop center.
        output_size (tuple): A tuple (width, height) for the output image.

    Returns:
        PIL.Image.Image or None: The cropped image, or None if the original can't be opened.
    """
    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        print(f"Warning: Source image not found at {image_path}")
        return None
    except Exception as e:
        print(f"Warning: Could not open image {image_path}. Error: {e}")
        return None

    img_width, img_height = img.size
    crop_width, crop_height = output_size

    # Calculate the ideal top-left corner of the crop box
    left = center_x - (crop_width // 2)
    top = center_y - (crop_height // 2)

    # --- Adaptive Shifting Logic ---
    # Adjust the left coordinate if the box exceeds the image boundaries
    # If the box is too far left, shift it right to start at 0
    if left < 0:
        left = 0
    # If the box is too far right, shift it left to end at the image width
    if left + crop_width > img_width:
        left = img_width - crop_width

    # Adjust the top coordinate similarly
    # If the box is too high, shift it down to start at 0
    if top < 0:
        top = 0
    # If the box is too low, shift it up to end at the image height
    if top + crop_height > img_height:
        top = img_height - crop_height

    # The above logic assumes the image is larger than the crop size.
    # If the image is smaller, we must clamp at 0 and then pad.
    if img_width < crop_width:
        left = 0
    if img_height < crop_height:
        top = 0

    # Define the final crop box
    right = left + crop_width
    bottom = top + crop_height

    # Crop the image using the final, adjusted coordinates
    cropped_img = img.crop((left, top, right, bottom))

    # If the original image was smaller than the output size, the cropped image
    # will be smaller. In this case, we pad it to the target size.
    if cropped_img.size != output_size:
        # Create a new black image of the target size
        # Ensure the mode matches the original image's mode (e.g., 'RGB', 'L')
        padded_img = Image.new(img.mode, output_size, 0)
        # Paste the cropped (smaller) image onto the center of the black background
        paste_x = (output_size[0] - cropped_img.width) // 2
        paste_y = (output_size[1] - cropped_img.height) // 2
        padded_img.paste(cropped_img, (paste_x, paste_y))
        return padded_img

    return cropped_img


# --- 1. DEFINE PATHS ---

# Input paths
SRC_IMAGE_DIR = "base"
CLASSIFICATIONS_CSV = "classification/classifications.csv"

# Output paths
CROPPED_IMAGE_DIR = "images"
OUTPUT_CSV = "all.csv"

# --- 2. SETUP ---
# Create output directory if it doesn't exist
if not os.path.exists(CROPPED_IMAGE_DIR):
    os.makedirs(CROPPED_IMAGE_DIR)

# Load the classifications CSV file
print(f"Loading classifications from: {CLASSIFICATIONS_CSV}")
try:
    df = pd.read_csv(CLASSIFICATIONS_CSV)
except FileNotFoundError:
    print(f"Error: Input CSV not found at {CLASSIFICATIONS_CSV}. Please check the path.")
    exit()

# List to store data for the new CSV
new_csv_data = []

# --- 3. PROCESSING LOOP ---
print("Starting image cropping process...")
# Using tqdm for a progress bar
for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing cells"):
    original_filename = row['image_filename']
    image_id = row['image_id']
    cell_id = row['cell_id']
    bethesda_system = row['bethesda_system']

    # Ensure coordinates are integers
    try:
        nucleus_x = int(row['nucleus_x'])
        nucleus_y = int(row['nucleus_y'])
    except ValueError:
        print(f"Warning: Skipping row {index} due to invalid coordinates for {original_filename}")
        continue

    # Construct the full path to the source image
    src_image_path = SRC_IMAGE_DIR + "/" + original_filename

    # Crop the image
    cropped_image = crop_and_pad_image(src_image_path, nucleus_x, nucleus_y)

    # If cropping was successful, save the image and record its data
    if cropped_image:
        # Construct the new filename (e.g., "original_name_1.png")
        base_name, _ = os.path.splitext(original_filename)
        new_filename = f"{base_name}_{cell_id}.png"

        # Construct the full destination path
        dest_image_path = CROPPED_IMAGE_DIR + "/" + new_filename

        # Save the cropped image
        cropped_image.save(dest_image_path)

        # Append data for the new CSV
        new_csv_data.append({
            'image_id': image_id,
            'image_name': new_filename,
            'class_name': bethesda_system
        })

# --- 4. CREATE NEW CSV ---
print("\nCreating new train.csv file...")
# Create a new DataFrame from the collected data
new_df = pd.DataFrame(new_csv_data)

# Save the new DataFrame to train.csv
new_df.to_csv(OUTPUT_CSV, index=False)

print(f"Processing complete!")
print(f"Cropped images saved to: {CROPPED_IMAGE_DIR}")
print(f"New metadata saved to: {OUTPUT_CSV}")
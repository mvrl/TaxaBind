#Converts all image samples to .jpg

from PIL import Image
import os
from tqdm import tqdm

def convert_to_jpg(input_file, output_file):
    try:
        with Image.open(input_file) as img:
            img.convert("RGB").save(output_file, "JPEG")
    except Exception as e:
        print(f"Conversion failed for {input_file}: {e}")

def main(input_directory, output_directory):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Traverse through the input directory and convert images
    for root, _, files in os.walk(input_directory):
        for file in tqdm(files):
            input_file = os.path.join(root, file)
            # Check if file extension is JPEG or jpg
            if file.lower().endswith(('.jpeg', '.jpg')):
                output_file = os.path.join(output_directory, os.path.splitext(file)[0] + ".jpg")
                convert_to_jpg(input_file, output_file)

if __name__ == "__main__":
    input_directory = "/storage1/fs1/jacobsn/Active/proj_smart/inat_image_sounds/images"
    output_directory = "/storage1/fs1/jacobsn/Active/proj_smart/inat_image_sounds/images_jpg"
    main(input_directory, output_directory)

import os
from PIL import Image

def downscale_and_save(input_dir, output_dir, target_size=(256, 256)):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize counter
    counter = 1

    # Walk through the input directory recursively
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            # Check if the file is an image
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                input_path = os.path.join(root, file)
                output_filename = f"{counter:06d}.png"  # Use 6-digit zero-padded numbers as filenames
                output_path = os.path.join(output_dir, output_filename)
                
                # Open the image
                with Image.open(input_path) as img:
                    # Resize the image
                    resized_img = img.resize(target_size, Image.LANCZOS)
                    # Save the resized image
                    resized_img.save(output_path)

                # Increment counter
                counter += 1

print("specify path")
return 0
# Example usage:
# input_directory = ""
# output_directory = ""
# downscale_and_save(input_directory, output_directory)

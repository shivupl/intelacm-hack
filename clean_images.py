from rembg import remove
from PIL import Image
import os

input_folder = "./fake-cars-ds"
output_folder = "./car_images_no_bg"

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):

    if filename.lower().endswith((".png", ".jpg")):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        with open(input_path, "rb") as inp:
            input_bytes = inp.read()
            output_bytes = remove(input_bytes)

        with open(output_path, "wb") as out:
            out.write(output_bytes)

print("success", output_folder)

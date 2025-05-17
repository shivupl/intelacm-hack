from rembg import remove
from PIL import Image
import os


def process_image(image, view):
    clean_img = remove_background(image)

    # run model


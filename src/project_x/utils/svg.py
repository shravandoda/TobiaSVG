import io

import cairosvg
from PIL import Image


def svg2pil(svg_text):
    png_bytes = cairosvg.svg2png(bytestring=svg_text.encode("utf-8"))
    image_data = io.BytesIO(png_bytes)
    pil_img = Image.open(image_data)
    pil_img.load()
    return pil_img

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

font_path = os.path.join(os.path.dirname(__file__), '..', 'fonts', 'DejaVuSansMono.ttf')

def rasterizer(text: str,
               font_path: str = font_path,
               font_size: int = 20,
               inverted: bool = False,
               max_size: int = 256,
               spaced: bool = True) -> np.ndarray: 
    font = ImageFont.truetype(font_path, font_size)
    text_color = 0 if inverted else 255
    bg_color = 255 if inverted else 0

    lines = text.split('\n')
    bbox = font.getbbox('A')  # Get bounding box for a single character
    char_width = bbox[2] - bbox[0] - spaced * 2
    char_height = bbox[3] - bbox[1]
    
    line_spacing = int(char_height * 0.3) if spaced else 0  # Espaciado vertical extra
    max_line_length = max(len(line) for line in lines)
    
    img_width = char_width * max_line_length
    img_height = (char_height + line_spacing) * len(lines) - line_spacing  # Evita agregar espacio extra después de la última línea

    img = Image.new('L', (img_width, img_height), color=bg_color)
    draw = ImageDraw.Draw(img)

    for row, line in enumerate(lines):
        for col, char in enumerate(line):
            x = col * char_width
            y = row * (char_height + line_spacing)
            draw.text((x, y), char, font=font, fill=text_color)

    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

    return np.array(img)

def save_image(img: np.ndarray, path: str):
    img = Image.fromarray(img)
    img.save(path)
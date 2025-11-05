import pyfiglet
from utils import rasterizer
import os

abs_path = os.path.dirname(os.path.abspath(__file__))

toxic_phrases_path = os.path.join(abs_path, 'toxic_phrases.csv')
safe_phrases_path = os.path.join(abs_path, 'safe_phrases.csv')
fonts_path = os.path.join(abs_path, 'fonts.csv')


toxic_phrases = []
safe_phrases = []
pyfiglet_fonts = []

with open(toxic_phrases_path, 'r') as file:
    line = file.read().strip()
    toxic_phrases = [phrase.strip().strip('"') for phrase in line.split(',')]

with open(fonts_path, 'r') as file:
    line = file.read().strip()
    pyfiglet_fonts = [font.strip().strip('"') for font in line.split(',')]

with open(safe_phrases_path, 'r') as file:
    line = file.read().strip()
    safe_phrases = [phrase.strip().strip('"') for phrase in line.split(',')]

print(f"Loaded {len(toxic_phrases)} toxic phrases.")
print(f"Loaded {len(safe_phrases)} safe phrases.")
print(f"Loaded {len(pyfiglet_fonts)} fonts.")

#Validate fonts
for font in pyfiglet_fonts:
    try:
        font_ = pyfiglet.FigletFont.getFont(font)
    except:
        pyfiglet_fonts.remove(font)

print(f"Validated {len(pyfiglet_fonts)} fonts.")

def generate_dataset():
    dataset_path = os.path.join(abs_path, 'dataset')
    imgs_path = os.path.join(dataset_path, 'imgs')
    
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    if not os.path.exists(imgs_path):
        os.makedirs(imgs_path)
    
    count = 0
    for phrase in toxic_phrases + safe_phrases:
        label = 'toxic' if phrase in toxic_phrases else 'safe'
        for font in pyfiglet_fonts:
            figlet = pyfiglet.Figlet(font=font)
            rendered_text = figlet.renderText(phrase)
            img_array = rasterizer(rendered_text, font_size=20, inverted=False, max_size=256, spaced=True)
            img_filename = f"{label}_{count}.png"
            img_path = os.path.join(dataset_path, img_filename)
            from utils import save_image
            save_image(img_array, img_path)
            count += 1
    print(f"Generated dataset with {count} images.")
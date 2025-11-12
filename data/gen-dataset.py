import pyfiglet
from utils import rasterizer, save_image
import os
import pandas as pd

abs_path = os.path.dirname(os.path.abspath(__file__))

toxic_phrases_path = os.path.join(abs_path, 'toxic_phrases.csv')
safe_phrases_path = os.path.join(abs_path, 'safe_phrases.csv')
fonts_path = os.path.join(abs_path, 'fonts.csv')
valid_fonts_path = os.path.join(abs_path, 'valid_fonts.csv')
invalid_fonts_path = os.path.join(abs_path, 'invalid_fonts.csv')

toxic_phrases = []
safe_phrases = []
pyfiglet_fonts = []
invalid_fonts = []
valid_fonts = []
worst_valid_font_idxs = {4,9,11,15,17,25,26,37,47,65,84,85,86,87,89,91,92,93,94,95,96,97,98,99,104,108,109,110,111,114,115,125,126,127,132,181,185,194,218}

with open(toxic_phrases_path, 'r') as file:
    line = file.read().strip()
    toxic_phrases = [phrase.strip().strip('"') for phrase in line.split(',')]

with open(safe_phrases_path, 'r') as file:
    line = file.read().strip()
    safe_phrases = [phrase.strip().strip('"') for phrase in line.split(',')]

with open(invalid_fonts_path, 'r') as file:
    line = file.read().strip()
    invalid_fonts = [font.strip().strip('"') for font in line.split(',')]

with open(fonts_path, 'r') as file:
    line = file.read().strip()
    pyfiglet_fonts = [font.strip().strip('"') for font in line.split(',')]

valid_fonts = [font for font in pyfiglet_fonts if font not in invalid_fonts]

print(f"Loaded {len(toxic_phrases)} toxic phrases.")
print(f"Loaded {len(safe_phrases)} safe phrases.")
print(f"Loaded {len(pyfiglet_fonts)} TOXASCII fonts.")
print(f"Loaded {len(invalid_fonts)} invalid TOXASCII fonts.")
print(f"Loaded {len(valid_fonts)} valid TOXASCII fonts.")

def generate_dataset():
    
    dataset = pd.DataFrame(columns=['image_path', 'text_label', 'caption', 'binary_target', 'font'])
    dataset_path = os.path.join(abs_path, 'dataset')
    imgs_path = os.path.join(dataset_path, 'imgs')
    
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    else:
        if os.path.exists(imgs_path):
            for f in os.listdir(imgs_path):
                os.remove(os.path.join(imgs_path, f))
        else:
            os.makedirs(imgs_path)
    
    count = 0
    for phrase in toxic_phrases + safe_phrases:
        caption = "An image containing toxic text" if phrase in toxic_phrases else "An image containing safe text"
        for font in valid_fonts:
            if valid_fonts.index(font) in worst_valid_font_idxs:
                continue
            ascii_art = pyfiglet.figlet_format(phrase, font=font)
            try:
                img = rasterizer(ascii_art, inverted=True)
                img_path = os.path.join(imgs_path, f'img_{count}.png')
                save_image(img, img_path)
                dataset.loc[len(dataset)] = [img_path, phrase, caption, 1 if phrase in toxic_phrases else 0, font]
                count += 1
            except:
                print(f"Failed to generate image for phrase: {phrase} and font: {font}")
                continue
    print(f"""Generated dataset with:
           - {count} images
           - {len(valid_fonts) - len(worst_valid_font_idxs)} fonts
           - {len(toxic_phrases)} toxic phrases
           - {len(safe_phrases)} safe phrases""")
    dataset.to_csv(os.path.join(abs_path, 'dataset.csv'), index=False)


if __name__ == "__main__":
    generate_dataset()
import os

abs_path = os.path.dirname(os.path.abspath(__file__))

toxic_phrases_path = os.path.join(abs_path, 'toxic_phrases.csv')
fonts_path = os.path.join(abs_path, 'fonts.csv')

toxic_phrases = []
pyfiglet_fonts = []

with open(toxic_phrases_path, 'r') as file:
    line = file.read().strip()
    toxic_phrases = [phrase.strip().strip('"') for phrase in line.split(',')]

with open(fonts_path, 'r') as file:
    line = file.read().strip()
    pyfiglet_fonts = [font.strip().strip('"') for font in line.split(',')]

import pyfiglet

not_found_fonts = []

for font in pyfiglet_fonts:
    try:
        string = pyfiglet.figlet_format(font, font=font)
    except:
        not_found_fonts.append(font)

print(not_found_fonts)
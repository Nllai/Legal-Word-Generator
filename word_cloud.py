import os
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_gradient_magnitude

from wordcloud import WordCloud, ImageColorGenerator, STOPWORDS

stopwords = set(STOPWORDS)
stopwords.add("said")
stopwords.add("defendant")
stopwords.add("plaintiff")
stopwords.add("V")
stopwords.add("S")
stopwords.add("claim")
stopwords.add("order")
stopwords.add("U")
stopwords.add("law")
stopwords.add("Affirmed")
stopwords.add("case")
stopwords.add("Held")
stopwords.add("Circuit")
stopwords.add("State")
stopwords.add("suit")
stopwords.add("lawsuit")
stopwords.add("court")
stopwords.add("Reversed")
stopwords.add("dismissal")
stopwords.add("whether")


def generate_word_cloud(text_path, image_path):
    filename = os.path.basename(text_path).split('.')[0]

    # Load text from file path
    text = open(text_path, encoding="utf-8").read()

    # load image for background.
    background_color = np.array(Image.open(image_path))
    background_color = background_color[::3, ::3]

    background_mask = background_color.copy()
    background_mask[background_mask.sum(axis=2) == 0] = 255

    edges = np.mean(
        [gaussian_gradient_magnitude(
            background_color[:, :, i] / 255., 2) for i in range(3)],
        axis=0)

    background_mask[edges > .08] = 255

    # Create word cloud
    wc = WordCloud(max_words=2000,
                   mask=background_mask,
                   max_font_size=40,
                   random_state=0,
                   relative_scaling=1,
                   stopwords=stopwords)

    wc.generate(text)

    apply_mask_color = ImageColorGenerator(background_color)
    wc.recolor(color_func=apply_mask_color)

    svg_text = wc.to_svg("pigs.svg")
    with open(f'output/{filename}.svg', 'w') as f:
        f.write(svg_text)


if __name__ == '__main__':
    # Generate wordcloud for three time periods with different backgrounds
    generate_word_cloud('texts/2017_to_2018.txt', 'background/1.png')
    generate_word_cloud('texts/2018_to_2019.txt', 'background/2.png')
    generate_word_cloud('texts/2019_to_2020.txt', 'background/2.png')


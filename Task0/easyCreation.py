import os
import shutil
import cv2
import random
import numpy as np

# Setting parameters
LABEL_COUNT = 50
LABEL_FREQUENCY = 200
FONT_STYLE = cv2.FONT_HERSHEY_DUPLEX
FONT_SIZE = 3
FONT_THICKNESS = 2
FONT_COLOR = (0, 0, 0)
IMAGE_HEIGHT = 400
IMAGE_WIDTH = 900

# Creating the output directory
output_dir = './Task0/EasySet'
os.makedirs(output_dir, exist_ok=True)
for file in os.listdir(output_dir):
    file_path = os.path.join(output_dir, file)
    if os.path.isdir(file_path):
        shutil.rmtree(file_path)
    else:
        os.remove(file_path)

# Creating a word list
words = [''.join(np.random.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), np.random.randint(5, 10))) for _ in range(LABEL_COUNT)]

# Creating the images
for word in words:
    word_dir = os.path.join(output_dir, word)
    os.makedirs(word_dir, exist_ok=True)

    (wordWidth, wordHeight), baseline = cv2.getTextSize(word, FONT_STYLE, FONT_SIZE, FONT_THICKNESS)

    for i in range(LABEL_FREQUENCY):
        # Creating a blank canvas (solid white colour)
        image = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 3), np.uint8) * 255

        # Positioning the text
        max_x = IMAGE_WIDTH - wordWidth
        max_y = IMAGE_HEIGHT - wordHeight
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
        position = (x, y)

        # Adding the text to the image
        cv2.putText(image, word, position, FONT_STYLE, FONT_SIZE, FONT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

        # Saving the image
        cv2.imwrite(os.path.join(word_dir, f'{word}-{i}.png'), image)

print(f'{LABEL_COUNT*LABEL_FREQUENCY} images created successfully at {output_dir}')
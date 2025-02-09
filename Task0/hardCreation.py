import os
import cv2
import shutil
import random
import numpy as np

# Setting parameters
LABEL_COUNT = 50
LABEL_FREQUENCY = 200
FONT_STYLE = [
    cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_COMPLEX_SMALL,
]
FONT_SIZE = [2.5, 3, 3.5, 4]
FONT_THICKNESS = [1, 2, 3, 4, 5]
IMAGE_HEIGHT = 400
IMAGE_WIDTH = 900

# Creating the output directory
output_dir = './Task0/HardSet'
os.makedirs(output_dir, exist_ok=True)
for file in os.listdir(output_dir):
    file_path = os.path.join(output_dir, file)
    if os.path.isdir(file_path):
        shutil.rmtree(file_path)
    else:
        os.remove(file_path)

# Creating a word list
words = [''.join(np.random.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'), np.random.randint(5, 10))) for _ in range(LABEL_COUNT)]

# Creating the images
for word in words:
    word_dir = os.path.join(output_dir, word)
    os.makedirs(word_dir, exist_ok=True)
    
    for i in range(LABEL_FREQUENCY):
        text = word

        # Creating a blank canvas (solid white colour)
        image = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 3), np.uint8) * 255
        noise = np.random.randint(0, 256, (IMAGE_HEIGHT, IMAGE_WIDTH, 3), np.uint8)
        image = cv2.addWeighted(image, 0.85, noise, 0.15, 0)

        # Setting text parameters
        style = np.random.choice(FONT_STYLE)
        size = np.random.choice(FONT_SIZE)
        thickness = np.random.choice(FONT_THICKNESS)
        while True:
            font_color = tuple([int(ele) for ele in np.random.randint(0, 256, 3)])
            if sum(font_color) < 500:
                break

        # Positioning the text
        (text_width, text_height), baseline = cv2.getTextSize(text, style, size, thickness)
        max_x = IMAGE_WIDTH - text_width
        max_y = IMAGE_HEIGHT - text_height
        x = random.randint(0, max_x)
        y = random.randint(text_height, max_y + text_height)
        position = (x, y)

        # Adding the text to the image
        cv2.putText(image, text, position, style, size, font_color, thickness, cv2.LINE_AA)

        # Saving the image
        cv2.imwrite(os.path.join(word_dir, f'{word}-{i}.png'), image)

print(f'{LABEL_COUNT*LABEL_FREQUENCY} images created successfully at {output_dir}')
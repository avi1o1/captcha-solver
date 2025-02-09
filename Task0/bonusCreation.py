import os
import cv2
import shutil
import random
import numpy as np

# Setting parameters
LABEL_COUNT = 20
LABEL_FREQUENCY = 10
FONT_STYLE = [
    cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_COMPLEX_SMALL,
]
FONT_SIZE = [2.5, 3, 3.5, 4]
FONT_THICKNESS = [1, 2, 3, 4, 5]
IMAGE_HEIGHT = 400
IMAGE_WIDTH = 900

# Creating the output directory
output_dir = './Task0/BonusSet'
os.makedirs(output_dir, exist_ok=True)
for file in os.listdir(output_dir):
    file_path = os.path.join(output_dir, file)
    if os.path.isdir(file_path):
        shutil.rmtree(file_path)
    else:
        os.remove(file_path)

# Creating a word list
words = [''.join(np.random.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'), np.random.randint(5, 10))) for _ in range(LABEL_COUNT)]

# Helper function to ensure the font color is not too similar to the background color
def is_color_similar(color, background_color, threshold=300):
    return sum(abs(c1 - c2) for c1, c2 in zip(color, background_color)) < threshold

# Creating the images
for word in words:
    word_dir = os.path.join(output_dir, word)
    os.makedirs(word_dir, exist_ok=True)
    
    for i in range(LABEL_FREQUENCY):
        text = word

        # Creating a blank canvas (green/red colour)
        redImage = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), np.uint8)
        greenImage = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), np.uint8)
        redImage[:, :, 2] = 255
        greenImage[:, :, 1] = 255

        noise = np.random.randint(0, 256, (IMAGE_HEIGHT, IMAGE_WIDTH, 3), np.uint8)
        redImage = cv2.addWeighted(redImage, 0.85, noise, 0.15, 0)
        greenImage = cv2.addWeighted(greenImage, 0.85, noise, 0.15, 0)

        # Setting text parameters
        style = np.random.choice(FONT_STYLE)
        size = np.random.choice(FONT_SIZE)
        thickness = np.random.choice(FONT_THICKNESS)
        while True:
            font_color = tuple([int(ele) for ele in np.random.randint(0, 256, 3)])
            if sum(font_color) > 100 and not is_color_similar(font_color, (0, 0, 255)) and not is_color_similar(font_color, (0, 255, 0)):
                break

        # Positioning the text
        (text_width, text_height), baseline = cv2.getTextSize(text, style, size, thickness)
        max_x = IMAGE_WIDTH - text_width
        max_y = IMAGE_HEIGHT - text_height
        x = random.randint(0, max_x)
        y = random.randint(text_height, max_y + text_height)
        position = (x, y)

        # Adding the text to the image
        cv2.putText(redImage, text[::-1], position, style, size, font_color, thickness, cv2.LINE_AA)
        cv2.putText(greenImage, text, position, style, size, font_color, thickness, cv2.LINE_AA)

        # Saving the image
        cv2.imwrite(os.path.join(word_dir, f'{word}-{i}-red.png'), redImage)
        cv2.imwrite(os.path.join(word_dir, f'{word}-{i}-green.png'), greenImage)

print(f'{LABEL_COUNT*LABEL_FREQUENCY*2} images created successfully at {output_dir}')
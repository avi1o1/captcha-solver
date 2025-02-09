import os
import shutil
import cv2
import numpy as np

# Setting parameters
IMAGE_COUNT = 5000
FONT_SIZE = 3
FONT_THICKNESS = 2
FONT_COLOR = (0, 0, 0)
IMAGE_HEIGHT = 400
IMAGE_WIDTH = 900
NOISE_FACTOR = 0.2
NOISE_MEAN = 0
NOISE_STD = 25

# Add new parameters
FONT_STYLES = [
    cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_COMPLEX_SMALL,
]
BACKGROUND_PATTERNS = ['grid', 'waves', 'random']

def add_complex_noise(image):
    # Background pattern
    pattern_type = np.random.choice(BACKGROUND_PATTERNS)
    if pattern_type == 'grid':
        spacing = np.random.randint(20, 40)
        for x in range(0, IMAGE_WIDTH, spacing):
            cv2.line(image, (x, 0), (x, IMAGE_HEIGHT), 
                    np.random.randint(200, 255, 3).tolist(), 1)
        for y in range(0, IMAGE_HEIGHT, spacing):
            cv2.line(image, (0, y), (IMAGE_WIDTH, y), 
                    np.random.randint(200, 255, 3).tolist(), 1)
    
    # Add more random elements
    num_shapes = np.random.randint(5, 25)
    for _ in range(num_shapes):
        shape_type = np.random.choice(['line', 'circle', 'rectangle'])
        color = np.random.randint(0, 255, 3).tolist()
        if shape_type == 'line':
            pt1 = (np.random.randint(0, IMAGE_WIDTH), 
                   np.random.randint(0, IMAGE_HEIGHT))
            pt2 = (np.random.randint(0, IMAGE_WIDTH), 
                   np.random.randint(0, IMAGE_HEIGHT))
            cv2.line(image, pt1, pt2, color, 
                    np.random.randint(1, 3))
        elif shape_type == 'circle':
            center = (np.random.randint(0, IMAGE_WIDTH), 
                     np.random.randint(0, IMAGE_HEIGHT))
            radius = np.random.randint(1, 7)
            cv2.circle(image, center, radius, color, -1)
        else:
            pt1 = (np.random.randint(0, 3), 
                   np.random.randint(0, 3))
            pt2 = (np.random.randint(pt1[0], 10), 
                   np.random.randint(pt1[1], 10))
            cv2.rectangle(image, pt1, pt2, color, -1)
    
    return image

# Creating the output directory
output_dir = './Task2/TestSet'
os.makedirs(output_dir, exist_ok=True)
for file in os.listdir(output_dir):
    file_path = os.path.join(output_dir, file)
    if os.path.isdir(file_path):
        shutil.rmtree(file_path)
    else:
        os.remove(file_path)

# Creating a word list
words = [''.join(np.random.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'), np.random.randint(5, 10))) for _ in range(IMAGE_COUNT)]

# Creating the images
for word in words:
    fontStyle = np.random.choice(FONT_STYLES)
    (wordWidth, wordHeight), baseline = cv2.getTextSize(word, fontStyle, FONT_SIZE, FONT_THICKNESS)

    # Creating a blank canvas (solid white colour)
    image = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 3), np.uint8) * 255

    # Positioning the text
    position = ((IMAGE_WIDTH - wordWidth)//2, (IMAGE_HEIGHT - wordHeight)//2)
    
    # Adding the text to the image
    cv2.putText(image, word, position, fontStyle, FONT_SIZE, FONT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    # Add noise to the image
    image = add_complex_noise(image)
    
    # Saving the image
    cv2.imwrite(os.path.join(output_dir, f'{word}.png'), image)

print(f'{IMAGE_COUNT} images created successfully at {output_dir}')
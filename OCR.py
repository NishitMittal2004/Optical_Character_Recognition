import cv2
import easyocr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Read image
image_path = 'Resources/Mamundi.jpeg'
img = cv2.imread(image_path)

# Instantiate text detector
reader = easyocr.Reader(['en'], gpu=False)

# Detect text on image
text_ = reader.readtext(img)

threshold = 0.25
texts_to_save = []

# Extract texts and save them
for t_, t in enumerate(text_):
    bbox, text, score = t

    if score > threshold:
        # Append text to list
        texts_to_save.append(text)

# Convert list of texts to DataFrame
df = pd.DataFrame({'Extracted Text': texts_to_save})

# Save DataFrame to Excel
# df.head()
df.to_csv('mamundi.csv', index=False)
import cv2
import easyocr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Read image
image_path = 'Resources/Adhaar_ex.png'
img = cv2.imread(image_path)

# Instantiate text detector
reader = easyocr.Reader(['en'], gpu=False)

# Detect text on image
text_ = reader.readtext(img)

threshold = 0.25
texts_to_save = []

# Draw bounding box and text and collect texts
for t_, t in enumerate(text_):
    bbox, text, score = t

    if score > threshold:
        cv2.rectangle(img, bbox[0], bbox[2], (0, 255, 0), 5)
        cv2.putText(img, text, bbox[0], cv2.FONT_HERSHEY_COMPLEX, 0.65, (255, 0, 0), 2)

        # Append text to list
        texts_to_save.append(text)

# Convert list of texts to DataFrame
df = pd.DataFrame({'Extracted Text': texts_to_save})

# Save DataFrame to Excel
df.to_excel('extracted_text.xlsx', index=False)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

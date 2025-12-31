import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

model = load_model('model/lion_tiger_model.h5')

img = cv2.imread('data/test.jpg')
img_resized = cv2.resize(img, (150,150))
img_resized = img_resized / 255.0
img_resized = np.reshape(img_resized, (1,150,150,3))

result = model.predict(img_resized)

label = "Tiger" if result[0][0] > 0.5 else "Lion"
print("Prediction:", label)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Prediction: " + label)
plt.axis('off')
plt.savefig('output/prediction_result.png')
plt.show()

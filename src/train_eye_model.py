import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# dataset path
dataset_path = os.path.join(BASE_DIR, "data", "eye_dataset")

categories = ["closed", "open"]

data = []
labels = []

img_size = 24

# load images
for category in categories:

    category_path = os.path.join(dataset_path, category)
    label = categories.index(category)

    print(f"Loading {category} images...")

    for root, dirs, files in os.walk(category_path):

        for img in files:

            img_path = os.path.join(root, img)

            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                continue

            image = cv2.resize(image, (img_size, img_size))

            data.append(image)
            labels.append(label)

# convert to numpy arrays
data = np.array(data)
labels = np.array(labels)

# normalize images
data = data / 255.0

# reshape for CNN
data = data.reshape(-1, img_size, img_size, 1)

# convert labels to categorical
labels = to_categorical(labels, 2)

print("Total samples loaded:", len(data))
print("Data shape:", data.shape)            

# split dataset
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

# build CNN model
model = Sequential()

model.add(Conv2D(32, (3,3), activation="relu", input_shape=(24,24,1)))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64, (3,3), activation="relu"))
model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(2, activation="softmax"))

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# train model
model.fit(
    X_train,
    y_train,
    epochs=15,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# create models folder if not exists
model_dir = os.path.join(BASE_DIR, "models")
os.makedirs(model_dir, exist_ok=True)

# save model
model_path = os.path.join(model_dir, "eye_model.h5")
model.save(model_path)

print("Model saved at:", model_path)
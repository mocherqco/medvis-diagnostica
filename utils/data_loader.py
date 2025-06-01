import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_and_preprocess(data_dir, csv_path, target_size=(224, 224), augment=False, config=None):
    df = pd.read_csv(csv_path)
    images = []
    labels = []

    for idx, row in df.iterrows():
        img_path = f"{data_dir}/{row['filename']}"
        try:
            img = image.load_img(img_path, target_size=target_size)
            img_array = image.img_to_array(img)
            if augment and config:
                datagen = ImageDataGenerator(rotation_range=config['augmentation']['rotation_range'],
                                             horizontal_flip=config['augmentation']['horizontal_flip'])
                img_array = datagen.random_transform(img_array)
            images.append(img_array)
            labels.append(1 if row['label'] == 'Diseased' else 0)
        except Exception as e:
            print(f"Error loading {row['filename']}: {e}")

    images = np.array(images)
    labels = np.array(labels)

    return images, labels

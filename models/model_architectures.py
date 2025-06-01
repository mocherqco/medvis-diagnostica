import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(input_shape, architecture='EfficientNetB0', transfer_learning=True, freeze_base=True):
    if architecture == 'EfficientNetB0':
        base_model = tf.keras.applications.EfficientNetB0(input_shape=input_shape, include_top=False, weights='imagenet')
    elif architecture == 'ResNet50':
        base_model = tf.keras.applications.ResNet50(input_shape=input_shape, include_top=False, weights='imagenet')
    elif architecture == 'MobileNetV2':
        base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    else:
        raise ValueError("Unsupported architecture specified.")

    if transfer_learning:
        base_model.trainable = not freeze_base

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    # Intentional error: using deprecated 'lr' instead of 'learning_rate'
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model

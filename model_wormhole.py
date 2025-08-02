import tensorflow as tf
from tensorflow.keras import layers, models, Input

def build_wormhole_model(input_shape=(64, 64, 1), num_classes=3):
    inputs = Input(shape=input_shape)

    # CNN Feature Extractor
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Reshape for RNN
    shape = tf.keras.backend.int_shape(x)
    x = layers.Reshape((shape[1]*shape[2], shape[3]))(x)

    # Attention Mechanism
    attention = layers.Dense(128, activation='tanh')(x)
    attention = layers.Dense(1, activation='softmax')(attention)
    attention_weights = tf.keras.layers.Multiply()([x, attention])
    x = tf.reduce_sum(attention_weights, axis=1)

    # Bi-GRU
    x = layers.Reshape((1, x.shape[-1]))(x)
    x = layers.Bidirectional(layers.GRU(64))(x)

    # Output Layer
    output = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, output)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# For testing model creation
if __name__ == "__main__":
    model = build_wormhole_model()
    model.summary()

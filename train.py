import tensorflow as tf

def Train_Model(dataset):
    # Define the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(dataset.shape[1])),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu')
    ])

    # Compile the model
    model.compile(
        loss='mse',
        optimizer='adam'
    )
    # Display the model summary
    model.summary()

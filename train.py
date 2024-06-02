import tensorflow as tf

def Train_Model(dataset):
    # Define the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(dataset.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(dataset.shape[1], activation='linear')
    ])

    # Compile the model
    model.compile(
        loss='mse',
        optimizer='adam'
    )

    # Display the model summary
    model.summary()
    
    # Train the model
    model.fit(dataset, dataset, epochs=100, batch_size=32, validation_split=0.2)
    
    return model


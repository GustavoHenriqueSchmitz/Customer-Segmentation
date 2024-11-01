import tensorflow as tf

def Autoencoder(dataset):
    """
    Train an autoencoder model on the provided dataset

    Args:
        dataset: Loaded preprocessed dataset

    Returns:
        model: Return a trained autoencoder model
    """
    input_layer = tf.keras.layers.Input(shape=(dataset.shape[1],))
    encoder = tf.keras.layers.Dense(40, activation='relu')(input_layer)
    latent_space = tf.keras.layers.Dense(20, activation='relu', name="latent_space")(encoder)
    decoder = tf.keras.layers.Dense(dataset.shape[1], activation='linear')(latent_space)

    # Define the Functional model
    model = tf.keras.Model(inputs=input_layer, outputs=decoder)

    # Compile the model
    model.compile(
        loss='mse',
        optimizer='adam'
    )

    # Display the model summary
    print("================================ Model's Summary ================================")
    model.summary()
    print("=================================================================================")
    
    # Train the model
    model.fit(dataset, dataset, epochs=100, batch_size=100, validation_split=0.15)
    
    return model

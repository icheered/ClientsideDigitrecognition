import tensorflow as tf

# Load the model
model = tf.keras.models.load_model("mnist_model.keras")

# Save the model in H5 format
model.save("mnist_model_resaved.h5")

print("Model resaved as 'mnist_model_resaved.h5'")

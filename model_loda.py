import tensorflow as tf, os

# Define the model path
model_path = '/home/ch/projects/python/datum_ml/find_match_api/Weight_Mapping_NN.keras'

# Check if the file exists
if os.path.exists(model_path):
    print(f"Model file found: {model_path}")
else:
    raise ValueError(f"Model file not found: {model_path}")

# Load the model
try:
    model = tf.saved_model.load(model_path)
    print("Model loaded successfully")
except Exception as e:
    raise ValueError(f"Failed to load model: {e}")

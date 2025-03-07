import tensorflow as tf

# List available GPU devices
gpus = tf.config.list_logical_devices('GPU')
print("Available GPUs:", gpus)
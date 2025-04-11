import tensorflow as tf
import numpy as np
import cv2
from keras.layers import TFSMLayer

# Step 1: Load the TensorFlow YOLOv4-tiny Model using TFSMLayer
model = TFSMLayer("checkpoints/yolov4-tiny", call_endpoint='serving_default')

# Step 2: Load and Prepare an Image
image_path = r'C:\Users\NAVEEN\Downloads\img.png'
image = cv2.imread(image_path)
image = cv2.resize(image, (416, 416))
image = np.array(image, dtype=np.float32)
image = np.expand_dims(image, axis=0)
image = image / 255.0

# Step 3: Define the Adversarial Pattern Function
def create_adversarial_pattern(input_image, input_label, model):
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = model(input_image)
        loss = tf.keras.losses.sparse_categorical_crossentropy(input_label, prediction)
    gradient = tape.gradient(loss, input_image)
    signed_grad = tf.sign(gradient)
    return signed_grad

# Step 4: Generate the Adversarial Image
target_label = tf.constant([42])  # Change this to the label you want to target
adv_pattern = create_adversarial_pattern(image, target_label, model)
eps = 0.1
adv_image = image + eps * adv_pattern
adv_image = np.clip(adv_image, 0, 1)
adv_image = adv_image[0] * 255.0
adv_image = adv_image.astype(np.uint8)

# Step 5: Save the Adversarial Image
cv2.imwrite('adversarial_image.jpg', adv_image)

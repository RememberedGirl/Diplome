import cv2
import numpy as np
import tensorflow as tf

# Load video
cap = cv2.VideoCapture('video/input_video.mp4')

# Load the pre-trained neural network
model = tf.keras.models.load_model('background_subtraction_model.h5')

# Loop through the frames of the video
while True:
    # Read the next frame
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to match the input size of the neural network
    resized_frame = cv2.resize(frame, (224, 224))

    # Preprocess the frame for input to the neural network
    x = np.array([resized_frame])
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    # Use the neural network to predict the probability that each pixel is part of the background
    y = model.predict(x)[0]

    # Threshold the predictions to create a binary mask of the background pixels
    background_mask = y < 0.5

    # Invert the mask to create a binary mask of the foreground (static objects) pixels
    static_mask = np.invert(background_mask)

    # Apply the static mask to the original frame to extract the static objects
    static_objects = cv2.bitwise_and(frame, frame, mask=static_mask)

    # Display the result
    cv2.imshow('Static objects', static_objects)

    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()

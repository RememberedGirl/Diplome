import cv2
import numpy as np

# Open the video file
cap = cv2.VideoCapture('video/input_video.mp4')

# Define the number of frames to use for SVD decomposition
num_frames = 100

# Initialize the buffer to store the frames
buffer = np.zeros((num_frames, 480, 640, 3), dtype=np.float32)

# Initialize the frame index
frame_index = 0

# Process the video frame-by-frame
while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # Break the loop if the video has ended
    if not ret:
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Add the frame to the buffer
    buffer[frame_index] = gray_frame.astype(np.float32)

    # Increment the frame index
    frame_index = (frame_index + 1) % num_frames

    # Compute the SVD decomposition of the buffer
    U, s, Vt = np.linalg.svd(buffer, full_matrices=False)

    # Keep only the first singular value, which corresponds to the static part of the scene
    s[1:] = 0

    # Reconstruct the buffer using the modified singular values
    buffer_svd = U @ np.diag(s) @ Vt

    # Extract the background frame as the average of the buffer
    background = np.mean(buffer_svd, axis=0).astype(np.uint8)

    # Extract the foreground by subtracting the background from the current frame
    foreground = cv2.absdiff(gray_frame, background)

    # Threshold the foreground to obtain the foreground mask
    _, fg_mask = cv2.threshold(foreground, 50, 255, cv2.THRESH_BINARY)

    # Apply morphological operations to remove noise and fill holes in the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

    # Invert the mask to obtain the background
    bg_mask = cv2.bitwise_not(fg_mask)

    # Extract the background by subtracting the mask from the current frame
    background = cv2.bitwise_and(frame, frame, mask=bg_mask)

    # Show the original frame, foreground mask, and background
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Foreground Mask', fg_mask)
    cv2.imshow('Background', background)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()

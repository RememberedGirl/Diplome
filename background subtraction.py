import cv2

# Initialize the background subtractor
background_subtractor = cv2.createBackgroundSubtractorMOG2()

# Open the video file
cap = cv2.VideoCapture('video/input_video.mp4')

# Process the video frame-by-frame
while True:
    # Read a frame from the video
    ret,  frame = cap.read()

    # Break the loop if the video has ended
    if not ret:
        break

    # Apply the background subtraction method
    foreground_mask = background_subtractor.apply(frame)

    # Apply thresholding to the foreground mask to extract the static objects
    _, foreground_mask = cv2.threshold(foreground_mask, 128, 255, cv2.THRESH_BINARY)

    # Extract the background by inverting the foreground mask
    background_mask = cv2.bitwise_not(foreground_mask)

    # Show the original frame, foreground mask, and background mask
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Foreground Mask', foreground_mask)
    # cv2.imshow('Background Mask', background_mask)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()

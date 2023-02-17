import cv2

# Open the video file
cap = cv2.VideoCapture('video/input_video.mp4')

# Initialize the background subtractor with the GMM method
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

# Process the video frame-by-frame
while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # Break the loop if the video has ended
    if not ret:
        break

    # Apply the background subtractor to extract the foreground mask
    fg_mask = bg_subtractor.apply(frame)

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

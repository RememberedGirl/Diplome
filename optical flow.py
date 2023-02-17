import cv2

# Open the video file
cap = cv2.VideoCapture('video/input_video.mp4')

# Initialize the parameters for the optical flow method
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Read the first frame of the video
ret, old_frame = cap.read()

# Convert the first frame to grayscale
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Create a mask to represent the static objects
mask = None

# Process the video frame-by-frame
while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # Break the loop if the video has ended
    if not ret:
        break

    # Convert the current frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply the optical flow method to track the motion of the pixels in the video
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, None, None, **lk_params)

    # Select only the pixels that are not moving
    static_pixels = p1[st == 1]

    # Create a new mask to represent the static objects
    new_mask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    new_mask.fill(0)

    # Set the pixels corresponding to the static objects to 255 in the new mask
    for x, y in static_pixels:
        cv2.circle(new_mask, (x, y), 5, 255, -1)

    # Apply morphological operations to remove noise and fill holes in the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    new_mask = cv2.morphologyEx(new_mask, cv2.MORPH_CLOSE, kernel)

    # Set the mask as the current mask
    mask = new_mask

    # Extract the background by subtracting the mask from the current frame
    background = cv2.bitwise_and(frame, frame, mask=mask)

    # Show the original frame, mask, and background
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Mask', mask)
    cv2.imshow('Background', background)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Set the current frame as the old frame for the next iteration
    old_gray = frame_gray.copy()

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()

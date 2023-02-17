import cv2

# Open video file
video = cv2.VideoCapture('video/input_video.mp4')

# Read first frame
ret, prev_frame = video.read()

# Loop through video frames
while True:
    # Read current frame
    ret, curr_frame = video.read()

    # Break if no more frames
    if not ret:
        break

    # Compute frame difference
    frame_diff = cv2.absdiff(curr_frame, prev_frame)
    gray_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray_diff, 25, 255, cv2.THRESH_BINARY)[1]

    # Display result
    cv2.imshow('Original Frame', curr_frame)
    cv2.imshow('Frame Difference', thresh)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    # Update previous frame
    prev_frame = curr_frame

# Clean up
video.release()
cv2.destroyAllWindows()
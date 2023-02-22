import cv2

# Load video
cap = cv2.VideoCapture('video/input_video.mp4')

# Define the threshold for detecting static objects
threshold = 1.0

# Set up the initial frame
ret, frame = cap.read()
gray_old = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Loop through the remaining frames
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray_old = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_new = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Compute the optical flow
    flow = None
    flow = cv2.calcOpticalFlowFarneback(prev=gray_old,
                                          next=gray_new, flow=flow,
                                          pyr_scale=0.8, levels=15, winsize=5,
                                          iterations=10, poly_n=5, poly_sigma=0,
                                          flags=10)
    #calcOpticalFlowFarneback(gray_old, gray_new, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Compute the magnitude of the flow
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Threshold the magnitude to detect static objects
    static_mask = mag < threshold

    # Set the static pixels to white and the moving pixels to black
    flow[static_mask] = [255, 255]

    # Display the result
    cv2.imshow('Static objects', flow)

    # Set the current frame as the old frame for the next iteration
    gray_old = gray_new

    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()

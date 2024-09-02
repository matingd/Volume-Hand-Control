import cv2 as cv
import mediapipe as mp
import math

# List of IDs for the fingertips of thumb, index, middle, ring, and pinky fingers
tip_ids = [4, 8, 12, 16, 20]

# Class for hand detection and tracking
class handdetectors:
    def __init__(self, mode=False, maxhands=2, detectionCon=0.5, trackCon=0.5):
        # Initialize parameters for hand detection and tracking
        self.mode = mode  # Whether to detect static images or video frames
        self.maxhands = maxhands  # Maximum number of hands to detect
        self.detectionCon = detectionCon  # Minimum detection confidence threshold
        self.trackCon = trackCon  # Minimum tracking confidence threshold

        # Initialize MediaPipe's hands solution
        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands(static_image_mode=self.mode,
                                        max_num_hands=self.maxhands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.trackCon)
        # Initialize drawing utilities for drawing hand landmarks
        self.mpdraw = mp.solutions.drawing_utils
    
    # Method to find and draw hands in a frame
    def findhands(self, frame, draw=True):
        # Flip the frame horizontally for a mirror effect
        frame = cv.flip(frame, 1)
        # Convert the BGR frame to RGB as required by MediaPipe
        rgbframe = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # Process the frame to detect hands
        self.results = self.hands.process(rgbframe)
        
        # If hands are detected
        if self.results.multi_hand_landmarks:
            for handlmk in self.results.multi_hand_landmarks:
                # Draw landmarks and connections on the hands if draw is True
                if draw:
                    self.mpdraw.draw_landmarks(frame, handlmk, self.mphands.HAND_CONNECTIONS)
        # Return the processed frame
        return frame
    
    # Method to find the position of landmarks on a specific hand
    def findposition(self, frame, draw=True, handNo=0):
        # Initialize a list to store landmark positions
        self.lmlist = []
        # If hands are detected
        if self.results.multi_hand_landmarks:
            # Get the landmarks of the specified hand
            myhand = self.results.multi_hand_landmarks[handNo]
            # Iterate over the landmarks and store their positions
            for id, lm in enumerate(myhand.landmark):
                h, w, c = frame.shape  # Get the dimensions of the frame
                cx, cy = int(lm.x * w), int(lm.y * h)  # Convert normalized coordinates to pixel values
                self.lmlist.append([id, cx, cy])  # Append the landmark ID and coordinates to the list
                # Draw a circle on a specific landmark if draw is True
                if draw:
                    if id == 4:  # Landmark 4 corresponds to the tip of the thumb
                        cv.circle(frame, (cx, cy), 15, (255, 0, 255), -1)
        # Return the list of landmark positions
        return self.lmlist

    # Method to check which fingers are up
    def fingersUp(self):
        fingers = []  # List to store the state of each finger (up or down)
        # Check if the middle finger is above the wrist (gesture detection)
        if self.lmlist[12][2] < self.lmlist[0][2]:
            # Check if the thumb is on the left or right side of the hand
            if self.lmlist[2][1] < self.lmlist[5][1]:
                # Check if the thumb is up
                if self.lmlist[tip_ids[0]][1] < self.lmlist[tip_ids[0] - 1][1]:
                    fingers.append(1)  # Thumb is up
                else:
                    fingers.append(0)  # Thumb is down
            else:
                # Check if the thumb is up (inverted for left hand)
                if self.lmlist[tip_ids[0]][1] > self.lmlist[tip_ids[0] - 1][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            # Check the state of the other four fingers
            for i in range(1, 5):
                if self.lmlist[tip_ids[i]][2] < self.lmlist[tip_ids[i] - 2][2]:
                    fingers.append(1)  # Finger is up
                else:
                    fingers.append(0)  # Finger is down
            total_fingers = fingers.count(1)  # Count the number of fingers that are up
        else:
            # Check for the inverted case when middle finger is below the wrist
            if self.lmlist[2][1] > self.lmlist[5][1]:
                if self.lmlist[tip_ids[0]][1] > self.lmlist[tip_ids[0] - 1][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                if self.lmlist[tip_ids[0]][1] < self.lmlist[tip_ids[0] - 1][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            for i in range(1, 5):
                if self.lmlist[tip_ids[i]][2] > self.lmlist[tip_ids[i] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            total_fingers = fingers.count(1)
        return fingers  # Return the list of finger states

    # Method to find the distance between two landmarks
    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.lmlist[p1][1:]  # Get coordinates of the first landmark
        x2, y2 = self.lmlist[p2][1:]  # Get coordinates of the second landmark
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Calculate the midpoint between the two landmarks

        # Draw a line and circles at the landmarks and midpoint if draw is True
        if draw:
            cv.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv.circle(img, (x1, y1), r, (255, 0, 255), cv.FILLED)
            cv.circle(img, (x2, y2), r, (255, 0, 255), cv.FILLED)
            cv.circle(img, (cx, cy), r, (0, 0, 255), cv.FILLED)

        # Calculate the Euclidean distance between the two landmarks
        length = math.hypot(x2 - x1, y2 - y1)

        # Return the distance, the image, and the coordinates of the points
        return length, img, [x1, y1, x2, y2, cx, cy]

# Main function to run the hand detection module
def main():
    # Initialize the webcam
    webcam = cv.VideoCapture(0)
    # Initialize the hand detector
    detector = handdetectors()
    
    while True:
        # Capture a frame from the webcam
        success, frame = webcam.read()
        # If the frame is not read successfully, break the loop
        if not success:
            break
        
        # Start the TickMeter to measure FPS
        tm = cv.TickMeter()
        tm.start()

        # Detect hands in the frame
        frame = detector.findhands(frame)
        # Get the positions of hand landmarks
        lmlist = detector.findposition(frame)
        # If landmarks are detected, print the position of the thumb tip (landmark 4)
        if len(lmlist) != 0:
            print(lmlist[4])
        
        # Stop the TickMeter and calculate FPS
        tm.stop()
        # Display FPS on the frame
        cv.putText(frame, "FPS : " + str(int(tm.getFPS())), (40, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        # Show the frame with detected hands
        cv.imshow('frame', frame)
        # Break the loop if the user presses 'q'
        if cv.waitKey(1) & 0xff == ord('q'):
            break
    
    # Release the webcam and close all OpenCV windows
    cv.destroyAllWindows()
    webcam.release()

# Run the main function when the script is executed
if __name__ == '__main__':
    main()

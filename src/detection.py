import cv2

# Load pre-trained face detection model from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def start_detection():
    """Start webcam and detect faces in real-time."""
    cap = cv2.VideoCapture(0)  # open default camera
    
    if not cap.isOpened():
        print("❌ Error: Could not access the camera")
        return
    
    print("✅ Face Detection started. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break
        
        # Convert to grayscale (needed for Haar Cascade)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        
        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Show video feed with detections
        cv2.imshow("Vision Guard - Face Detection", frame)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

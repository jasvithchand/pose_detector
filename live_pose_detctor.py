import cv2 as cv
import mediapipe as mp

def count_raised_fingers(lm_list):
    if lm_list[4][1] < lm_list[3][1]:
        finger_count = 1
    else:
        finger_count = 0

    for i in range(5, 21, 4):
        if lm_list[i][2] < lm_list[i - 2][2]: 
            finger_count += 1

    return finger_count

def detect_and_draw_faces():

    front_face_cascade = cv.CascadeClassifier('classifiers/haar_face.xml')
    side_face_cascade = cv.CascadeClassifier('classifiers/haar_sideface.xml')

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)
    
    webcam = cv.VideoCapture(0)

    while True:
        ret, frame = webcam.read()

        if not ret:
            print("Error: Could not capture a frame from the webcam.")
            break

        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        faces = front_face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
        right_side = side_face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv.putText(frame, "Front", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        for (x, y, w, h) in right_side:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv.putText(frame, "Right Side", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        flipped_frame = cv.flip(gray_frame, 1)
        left_side = side_face_cascade.detectMultiScale(flipped_frame, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in left_side:
            x = flipped_frame.shape[1] - (x + w) 
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv.putText(frame, "Left Side", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                lm_list = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]

                finger_count = count_raised_fingers(lm_list)

                if finger_count == 2:
                    cv.putText(frame, "Pose Detected", (0,200), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)


        cv.imshow("Pose Detection", frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    webcam.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    detect_and_draw_faces()

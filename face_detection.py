import cv2
import os
import time
import sys

CONFIDENCE_THRESHOLD = 0.6
MODEL_DIR = "models"
SAVED_FACES_DIR = "saved_faces"




def load_model():
    

    base_path = os.path.dirname(os.path.abspath(__file__))

    prototxt_path = os.path.join(base_path, MODEL_DIR, "deploy.prototxt")
    model_path = os.path.join(base_path, MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

    if not os.path.exists(prototxt_path):
        raise FileNotFoundError(f"Prototxt file not found at {prototxt_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Caffe model not found at {model_path}")

    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    return net


def initialize_camera():
    
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Webcam not detected. Please check your camera connection.")

    return cap


def create_output_directory():
    
    base_path = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(base_path, SAVED_FACES_DIR)

    os.makedirs(save_path, exist_ok=True)
    return save_path


def detect_faces(net, frame):
    

    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )

    net.setInput(blob)
    detections = net.forward()

    return detections, h, w


def save_face(face_img, save_path, counter):
    
    filename = f"face_{counter}.jpg"
    full_path = os.path.join(save_path, filename)
    cv2.imwrite(full_path, face_img)


def main():
    try:
        print("[INFO] Loading face detection model...")
        net = load_model()

        print("[INFO] Initializing webcam...")
        cap = initialize_camera()

        save_path = create_output_directory()

        print("[INFO] Starting real-time face detection... Press 'q' to quit.")

        face_counter = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()

            if not ret:
                print("[WARNING] Failed to grab frame.")
                break

            detections, h, w = detect_faces(net, frame)

            face_count = 0

            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > CONFIDENCE_THRESHOLD:
                    face_count += 1

                    box = detections[0, 0, i, 3:7] * [w, h, w, h]
                    (startX, startY, endX, endY) = box.astype("int")

                    
                    startX, startY = max(0, startX), max(0, startY)
                    endX, endY = min(w - 1, endX), min(h - 1, endY)

                    face = frame[startY:endY, startX:endX]

                    if face.size > 0:
                        face_counter += 1
                        save_face(face, save_path, face_counter)

                    text = f"{confidence * 100:.2f}%"
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    cv2.putText(frame, text,
                                (startX, startY - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0),
                                2)

            
            elapsed_time = time.time() - start_time
            fps = face_counter / elapsed_time if elapsed_time > 0 else 0

            
            cv2.putText(frame, f"Faces: {face_count}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2)

            cv2.putText(frame, f"FPS: {fps:.2f}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 0, 0),
                        2)

            cv2.imshow("Real-Time Face Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Application closed successfully.")

    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    except RuntimeError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    except Exception as e:
        print(f"[UNEXPECTED ERROR] {e}")
        sys.exit(1)



if __name__ == "__main__":
    main()

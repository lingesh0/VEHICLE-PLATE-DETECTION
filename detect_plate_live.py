import cv2
import easyocr
import pandas as pd
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Initialize EasyOCR
reader = easyocr.Reader(['en'])

# Use mobile camera or laptop webcam
ip_camera_url = "http://192.168.1.8:8080/video"  # Mobile camera URL
cap = cv2.VideoCapture(ip_camera_url)  # Use 0 for laptop webcam

# Output CSV file
csv_filename = "detected_plates1.csv"
data = []
plate_count = 1

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to capture image. Check camera connection.")
        break  # Exit loop if camera fails

    # Detect plates
    results = model(frame)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
            plate_crop = frame[y1:y2, x1:x2]

            # Convert to grayscale
            gray_plate = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)

            # OCR to extract text
            text = reader.readtext(gray_plate, detail=0)
            plate_text = " ".join(text).strip()

            if plate_text:
                print(f"✔ Detected Plate: {plate_text}")
                data.append([plate_count, plate_text])
                plate_count += 1

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, plate_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show video
    cv2.imshow("License Plate Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save results to CSV
df = pd.DataFrame(data, columns=["S.No", "Number Plate"])
df.to_csv(csv_filename, index=False)
print(f"✅ Saved detected plates to {csv_filename}")

# Release resources
cap.release()
cv2.waitKey(1)  # Fix for OpenCV crash
cv2.destroyAllWindows()

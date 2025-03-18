from ultralytics import YOLO
import cv2
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from paddleocr import PaddleOCR

# Load a better YOLOv8 model (medium version for accuracy)
model = YOLO("yolov8m.pt")  # 'm' means medium version (better accuracy)

# Initialize PaddleOCR (English model)
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Dataset path
dataset_path = "Real-Time-License-Plate-Dataset/images"
output_csv = "detected_plates.csv"

# List all image files
image_files = [f for f in os.listdir(dataset_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Initialize results storage
plate_results = []

# Process each image in dataset
for image_file in tqdm(image_files, desc="Processing Images"):
    image_path = os.path.join(dataset_path, image_file)
    image = cv2.imread(image_path)

    # Run YOLO detection
    results = model(image)

    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)

            # Extract the license plate region
            plate = image[y1:y2, x1:x2]

            # Convert to grayscale and apply noise removal
            gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)  # Remove noise
            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

            # Use PaddleOCR for text extraction
            ocr_result = ocr.ocr(gray, cls=True)

            # Extract text if found
            plate_text = ""
            if ocr_result and ocr_result[0]:
                plate_text = "".join([word[1][0] for word in ocr_result[0]])

            # Store results
            plate_results.append([image_file, plate_text])
            print(f"Image: {image_file}, Detected Plate: {plate_text}")

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the image with detection
    cv2.imshow("License Plate Detection", image)
    cv2.waitKey(500)  # Show each image for 500ms

cv2.destroyAllWindows()

# Save results to CSV
df = pd.DataFrame(plate_results, columns=["Image Name", "Detected Plate"])
df.to_csv(output_csv, index=False)

print(f"\n✅ Detection complete! Results saved in {output_csv}")

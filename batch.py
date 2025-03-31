import cv2
import numpy as np
import imutils
import easyocr
import os
import paddleocr
from ultralytics import YOLO

def load_image(image_path):
    """Loads an image and converts it to grayscale."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray

def enhance_image(gray):
    """Applies denoising and contrast enhancement."""
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    equalized = cv2.equalizeHist(denoised)
    return equalized

def detect_edges(gray):
    """Applies Canny edge detection."""
    edged = cv2.Canny(gray, 50, 150)
    return edged

def find_license_plate_yolo(image_path):
    """Uses YOLO for license plate detection."""
    model = YOLO("yolov8n.pt")  # Load pre-trained model
    results = model(image_path)
    for result in results:
        if result.boxes:
            x1, y1, x2, y2 = result.boxes.xyxy[0].tolist()
            return int(x1), int(y1), int(x2), int(y2)
    return None

def crop_license_plate(image, bbox):
    """Crops the detected license plate from the image."""
    x1, y1, x2, y2 = bbox
    return image[y1:y2, x1:x2]

def recognize_text_easyocr(cropped_plate):
    """Recognizes text from the license plate using EasyOCR."""
    reader = easyocr.Reader(['en'])
    result = reader.readtext(cropped_plate)
    return result[0][-2] if result else "Not detected"

def recognize_text_paddleocr(cropped_plate):
    """Recognizes text from the license plate using PaddleOCR."""
    reader = paddleocr.OCR()
    result = reader.ocr(cropped_plate)
    return result[0][-1][0] if result else "Not detected"

def draw_results(image, bbox, plate_text, image_name):
    """Draws the detected license plate and recognized text on the image."""
    if bbox:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    cv2.imshow(f"License Plate - {image_name}", image)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

def process_images(folder_path):
    """Processes all images in a given folder."""
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("No images found in the folder.")
        return
    
    for image_name in image_files:
        image_path = os.path.join(folder_path, image_name)
        print(f"\nProcessing: {image_name}")
        
        image, gray = load_image(image_path)
        enhanced_gray = enhance_image(gray)
        bbox = find_license_plate_yolo(image_path)
        
        if bbox:
            cropped_plate = crop_license_plate(image, bbox)
            plate_text = recognize_text_easyocr(cropped_plate)
            print(f"Detected License Plate: {plate_text}")
            draw_results(image, bbox, plate_text, image_name)
        else:
            print("License plate not detected.")

if __name__ == "__main__":
    folder_path = "D:\\ML\\VehiclePlateDetection\\dataset\\Real Time Licence Plates Data\\Images"
    process_images(folder_path)

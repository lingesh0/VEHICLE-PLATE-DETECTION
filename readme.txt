🚗 Vehicle License Plate Detection
This project detects and recognizes vehicle license plates using YOLOv8. It includes scripts for live and image-based detection, along with a trained YOLOv8 model.

📂 Project Structure
pgsql
Copy
Edit
VehiclePlateDetection/
│── dataset/
│── Real-Time-License-Plate-Detection/
│   ├── Annotations/
│   ├── images/
│── venv/
│   ├── Include/
│   ├── Lib/
│   ├── Scripts/
│   ├── share/
│── car.jpg
│── detect_plate_live.py
│── detect_plate.py
│── detected_plates.csv
│── detected_plates1.csv
│── yolov8_training.py
│── yolov8m.pt
│── yolov8n.pt
📥 Download the Dataset
This project uses the Real-Time License Plate Dataset from Kaggle. Download it using the following steps:

Install the Kaggle API if you haven’t already:

bash
Copy
Edit
pip install kaggle
Download your Kaggle API key:

Go to Kaggle and sign in.
Click on your profile picture > Account.
Scroll down to API and click Create New API Token.
A file named kaggle.json will be downloaded.
Move kaggle.json to the appropriate location:

bash
Copy
Edit
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
Download the dataset:

bash
Copy
Edit
kaggle datasets download -d radhikakj/real-time-license-plate-dataset
Extract the dataset:

bash
Copy
Edit
unzip real-time-license-plate-dataset.zip -d dataset/
🚀 Setup Instructions
1️⃣ Install Dependencies
First, create and activate a virtual environment (recommended).

bash
Copy
Edit
python -m venv venv
source venv/Scripts/activate  # For Windows
# OR
source venv/bin/activate  # For Linux/Mac
Then, install the required libraries:

bash
Copy
Edit
pip install -r requirements.txt
If you don’t have requirements.txt, install the main dependencies manually:

bash
Copy
Edit
pip install ultralytics opencv-python numpy pandas torch torchvision torchaudio
2️⃣ Run Plate Detection on an Image
bash
Copy
Edit
python detect_plate.py --image_path car.jpg
3️⃣ Run Live Camera Detection
Ensure your webcam or an IP camera is connected, then run:

bash
Copy
Edit
python detect_plate_live.py
If using an IP camera, modify detect_plate_live.py to use your camera’s URL.

4️⃣ Training the YOLOv8 Model
If you want to train your own YOLOv8 model on a custom dataset, run:

bash
Copy
Edit
python yolov8_training.py
Make sure your dataset is properly structured inside the dataset/ folder.

🔍 Output
The detected plate numbers will be saved in detected_plates.csv.
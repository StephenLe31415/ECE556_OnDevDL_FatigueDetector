# PROJECT OVERVIEW #
This project implements a real-time drowsiness detection system using a Raspberry Pi and computer vision. The system monitors a user's eye state and mouth activity (yawning) to determine whether they are alert or drowsy.

## Pipeline ##
- Face detection using Haar cascades
- Region-of-interest extraction (eyes + mouth)
- Two lightweight CNN models:
    + Eye model (grayscale-based)
    + Mouth model (RGB-based)
- Real-time inference using TensorFlow Lit (INT8 quantized model)
- Decision logic to classify user state:
    + Alert
    + Drowsy (eyes closed OR yawning)

The system is optimized for edge development, achieving real-time performance on a Raspberry Pi with limited compute resources.

# PROJECT ANALYSIS #
1. Problem formulation:
- Driver drowsiness detection is framed as binary classification problem:
    + Eye state --> Open/Closed
    + Mouth state --> Yawn/No Yawn
- Final classification = (Eyes closed) OR (Yawning)

2. Dataset & Preprocessing:
- Eye model:
    + Trained on grayscale images
    + Input shape: (24, 24, 3) (grayscale expanded to RGB)

- Mouth model:
    + Trained on RGB images
    + Input shape: (24, 24, 3) (RGB)

- Key challenge: mismatch between training domain (RGB vs. grayscale) and deployment pipeline

- Solutoin: separate preprocessing pipelines
    + preprocess_eye() --> grayscale --> RGB
    + preprocess_mouth() --> RGB directly

3. Model architecture
- Both models use lightweight, custom CNNs:
Conv --> ReLU --> Pool
Conv --> ReLU --> Pool
conv --> ReLU --> Pool
Flatten --> Dense --> sigmoid

- Design goals:
    + Small memory footprint
    + Fast inference on CPU
    + Good generalization on low-resolution inputs (24 x 24)

4. Pruning and Optimization
- To improve edge performance:
    + Apply fine-grained weight pruning (~50%)
    + Removed low-magnitude weights
    + Fine-tune after pruning

5. Quantization (INT8)
- Models are converted to TensorFlow Lite INT8
    + Reduces model size
    + Speeds up inference
    + Enables efficient execution on Raspberry Pi

6. Deployment Pipeline:
Real-time loop:
    1) Capture frame through Pi Camera
    2) Convert to grayscale for detection
    3) Detect face (downsampled frame)
    4) Extract regions:
        + Eyes (upper face)
        + Mouth (lower face)
    5) Run inference (every N frame)
    6) Combine predictions --> Final decision

7. Key challenges & Solutions
- Problem 1: Model stuck at "No Yawn"
    + Cause: incorrect use of quantized outputs
    + Fix: dequantization using scale + zero_point

- Problem 2: Unstable mouth detection
    + Cause: inconsistent ROI cropping
    + Fix:
        1. Expand lower face region
        2. Use relative bound box scaling
        3. Clamp coordinates to valid ranges

- Problem 3: Manualing labeling at high FPS
    + Isuse: frame-level annotation too fast
    + Solution:
        1. Log predictions to CSV
        2. Use event-based labeling (time segments)

- Problem 4: Mixed input formats
    + Cause: Eye = grayscale but Mouth = RGB
    + Fix: separate preprocessing pipelines

8. Evaluation Metrics:
The system supports:
    + Overall accuracy
    + Precision
    + Recall
    + F1-score
    + False Alarm Rate (FAR)
--> Evaluation is performed using logged predictions (run_log.csv)

# PROJECT DEPLOYMENT INSTRUCTIONS #
1. Create a virtual environemnt in Python and '''cd''' into that venv

2. Install dependencies
'''pip install numpy opencv-python tflite-runtime matplotlib
'''

3. On Raspberry Pi:
'''sudo apt install python3-opencv
'''

4. Clone this GitHub repo on your Raspberry Pi

5. Run these commands in order:
- Train the eye detection model using "cew" dataset
'''
python eye_training.py
'''

- Train the mouth detection model using "yawn_dataset"
'''
python mouth_training.py
'''

- Run the deployment code on the Rasberry Pi for real-time detection
'''python deploy.py

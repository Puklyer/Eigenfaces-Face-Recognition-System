# Eigenfaces-Face-Recognition-System
A Python-based face recognition application implementing the Eigenfaces method with PCA (Principal Component Analysis) using multiple preprocessing techniques including Standard, Lanczos, and Class Representatives approaches.
Overview
This project provides an advanced face recognition system using dimensionality reduction techniques. The system projects facial images into a lower-dimensional "face space" using eigenfaces, enabling efficient face recognition with various preprocessing methods and distance metrics.
Features

Multiple Preprocessing Methods:

Standard PCA (Principal Component Analysis)
Lanczos Algorithm for efficient eigenvalue computation
Class Representatives method using representative images from each class


Multiple Distance Metrics: Euclidean, Manhattan, Infinity (Chebyshev), and Cosine similarity
Ghost Face Visualization: Display eigenface decomposition of test images
Interactive GUI: User-friendly interface built with Tkinter
Configurable k-value: Choose the number of principal components (20, 40, 60, 80, 100)
Performance Analysis: Comprehensive statistics including Recognition Rate (RR) and Average Query Time (AQT)
Batch Testing: Automated evaluation on test images
Data Export: Save performance metrics to CSV files

Requirements
python >= 3.7
numpy
opencv-python (cv2)
matplotlib
pillow
pandas
tkinter (usually included with Python)
Installation

Clone or download this repository
Install required packages:

bashpip install numpy opencv-python matplotlib pillow pandas

Download the AT&T Database of Faces (ORL Database) and place it in the project directory

Dataset Structure
The code expects the following folder structure:
ORL/
├── s1/
│   ├── 1.pgm
│   ├── 2.pgm
│   ├── ...
│   └── 10.pgm
├── s2/
│   ├── 1.pgm
│   ├── 2.pgm
│   ├── ...
│   └── 10.pgm
└── ...

40 people (folders s1 to s40)
10 images per person (.pgm format)
Images 1-8 are used for training
Images 9-10 are used for testing

Usage
Running the Application

Ensure the base_folder variable in the code points to your ORL dataset directory
Run the script:

bashpython eigenfaces_recognition.py
Using the GUI
Main Features:

Select k value: Choose the number of principal components (20, 40, 60, 80, or 100)
Select preprocessing type:

Standard: Classical PCA eigenface method
Lanczos: Efficient eigenvalue computation for large datasets


Update k: Apply the selected k value with chosen preprocessing method
Recognition with representatives: Test recognition using class representative images
Load image: Select and recognize a face from your test image
Display ghosts: Visualize eigenface decomposition of the loaded image
Run statistics: Generate comprehensive performance analysis

Workflow Example

Select k=100 and preprocessing type "Standard"
Click "Update k" to compute eigenfaces
Click "Load image" to test recognition on a face image
Click "Display ghosts" to see the eigenface components
Use "Run statistics" to evaluate performance across all methods and parameters

How It Works
Eigenfaces Method
The eigenfaces approach treats face recognition as a dimensionality reduction problem:

Training Phase:

Compute the mean face from training images
Center the data by subtracting the mean
Calculate eigenvectors (eigenfaces) of the covariance matrix
Keep only the top k eigenfaces with largest eigenvalues
Project all training images onto the eigenface space


Recognition Phase:

Center the test image by subtracting the mean face
Project the test image onto eigenface space
Find the closest match using distance metrics
Return the identified person



Preprocessing Methods
Standard PCA

Classical eigenface computation using full eigendecomposition
Best accuracy but computationally expensive for large datasets

Lanczos Algorithm

Efficient iterative method for computing top k eigenvectors
Faster preprocessing with minimal accuracy trade-off
Ideal for large-scale applications

Class Representatives

Uses one representative image per person instead of all training images
Reduces computational complexity significantly
Slightly lower accuracy but much faster processing

Ghost Faces
Ghost faces represent the contribution of each eigenface to reconstructing the test image. They help visualize which facial features are most important for recognition.
Performance Metrics

Recognition Rate (RR): Percentage of correctly identified faces (based on 80 test images)
Average Query Time (AQT): Average time to process one face recognition query
Preprocessing Time: Time required to compute eigenfaces for given k value

Results are automatically saved to:

RecognitionRateEig.csv: Recognition rates for all methods, k values, and norms
AQT_eig.csv: Average query times for all configurations

Configuration
Key parameters in the code:
pythonimage_width = 92          # Width of resized images
image_height = 112        # Height of resized images
num_people = 40           # Number of individuals in dataset
num_images_per_person = 8 # Training images per person
base_folder = 'ORL'       # Dataset folder path
Output Files

RecognitionRateEig.csv: Recognition rates organized by method, k value, and norm
AQT_eig.csv: Average query times for each configuration
Console output showing real-time progress during statistics computation

Mathematical Background
PCA (Principal Component Analysis)
Reduces high-dimensional face images (10,304 pixels) to k-dimensional vectors while preserving maximum variance.
Eigenfaces
The eigenvectors of the face covariance matrix, representing the principal directions of variation in face images.
Distance Metrics

Euclidean: L2 norm, standard straight-line distance in eigenspace
Manhattan: L1 norm, sum of absolute differences
Infinity: L∞ norm, maximum absolute difference
Cosine: Measures angle between vectors (1 - cosine similarity)

Advantages of Eigenfaces

Significant dimensionality reduction (from ~10,000 to ~100 dimensions)
Fast recognition after preprocessing
Robust to minor variations in lighting and pose
Computationally efficient for large databases

Limitations

Sensitive to major lighting changes
Poor performance with significant pose variations
Requires face alignment and normalization
Grayscale images only
All faces must be approximately same size

Future Improvements

Add Fisherfaces (LDA) for better class separation
Implement deep learning methods (CNN-based recognition)
Add real-time webcam recognition
Support for color images and multi-scale analysis
Robust face alignment preprocessing
GPU acceleration for faster computation
Cross-database evaluation

Troubleshooting
Common Issues:

"Could not load the image": Ensure image is in supported format (PGM, PNG, JPG, BMP)
"No image has been loaded": Load an image before displaying ghosts
Poor recognition: Try increasing k value or using different preprocessing method
Slow performance: Use Lanczos method or reduce k value

Research Applications
This implementation is suitable for:

Academic research in computer vision
Comparative studies of dimensionality reduction techniques
Educational purposes in machine learning courses
Benchmarking face recognition algorithms

References

Turk, M., & Pentland, A. (1991). "Eigenfaces for recognition". Journal of Cognitive Neuroscience.
AT&T Laboratories Cambridge: The Database of Faces (ORL Database)
Lanczos Algorithm for efficient eigenvalue computation

License
This project is provided as-is for educational and research purposes.
Acknowledgments

AT&T Laboratories Cambridge for the ORL face database
Scientific Python community: NumPy, OpenCV, Matplotlib
Turk and Pentland for the original Eigenfaces method

Contact
For questions, bug reports, or contributions, please open an issue in the repository.

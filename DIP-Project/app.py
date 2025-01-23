import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
import matplotlib.pyplot as plt

# Initialize the Flask app
app = Flask(__name__)

# Configure upload folders
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

def apply_gamma_correction(image, gamma):
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)


def histogram_equalization(image):
    if len(image.shape) == 2:  # Grayscale
        return cv2.equalizeHist(image)
    else:  # Color image
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

def laplacian_filter(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return cv2.convertScaleAbs(laplacian)

def sobel_operator(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobelx, sobely)
    return cv2.convertScaleAbs(sobel)

def apply_frequency_filter(image, filter_type, cutoff):
    rows, cols = image.shape[:2]
    crow, ccol = rows // 2, cols // 2

    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    mask = np.zeros((rows, cols, 2), np.float32)
    for i in range(rows):
        for j in range(cols):
            dist = np.sqrt((i - crow)**2 + (j - ccol)**2)
            if filter_type == 'lowpass' and dist <= cutoff:
                mask[i, j] = 1
            elif filter_type == 'highpass' and dist > cutoff:
                mask[i, j] = 1

    filtered_dft = dft_shift * mask
    dft_ishift = np.fft.ifftshift(filtered_dft)
    img_back = cv2.idft(dft_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    return cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file is uploaded
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Read the image
            image = cv2.imread(filepath)

            # Apply algorithms
            gamma_image = apply_gamma_correction(image, gamma=0.5)
            hist_image = histogram_equalization(image)
            laplacian_image = laplacian_filter(image)
            sobel_image = sobel_operator(image)
            lowpass_image = apply_frequency_filter(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 'lowpass', cutoff=50)
            highpass_image = apply_frequency_filter(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 'highpass', cutoff=50)


            # Save results
            results = {
                'gamma': gamma_image,
                'histogram': hist_image,
                'laplacian': laplacian_image,
                'sobel': sobel_image,
                'lowpass': lowpass_image,
                'highpass':highpass_image
            }

            result_paths = {}
            for name, result in results.items():
                result_path = os.path.join(app.config['RESULT_FOLDER'], f"{name}_{file.filename}")
                cv2.imwrite(result_path, result)
                result_paths[name] = result_path

            return render_template('index.html', uploaded_image=filepath, result_paths=result_paths)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)



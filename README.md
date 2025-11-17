# Deepfake Image Detection Project

This project uses a deep learning model (MobileNetV2) to detect whether an image is a real or a deepfake. It includes scripts for training the model, making predictions, and a Streamlit web application to interact with the model.

This version has been optimized for use on a T4 GPU in Google Colab and includes important bug fixes.

## Key Features & Optimizations

*   **Deep Learning Model:** Utilizes a fine-tuned MobileNetV2 model for efficient and accurate deepfake detection.
*   **T4 GPU Optimization:** The training script (`train.py`) is optimized for T4 GPUs by using **mixed-precision training**, which significantly speeds up training time.
*   **Bug Fixes:** Critical bugs in image preprocessing have been fixed across all scripts (`train.py`, `predict.py`, `app.py`, `utils.py`). The image size is now correctly set to `(128, 128)` and uses the appropriate `preprocess_input` function.
*   **Streamlit Web App:** An easy-to-use web interface (`app.py`) to upload an image and get a prediction. Model loading is cached for better performance.

## Project Structure

The project follows a simple structure:
```
/
├── app.py              # Streamlit web application
├── predict.py          # Script for making single predictions
├── train.py            # Script for training the model
├── utils.py            # Utility functions (not directly used by app.py or predict.py)
├── requirements.txt    # Python dependencies
├── deepfake_detection_model.h5   # The trained model (after running train.py)
└── real_and_fake_face/ # Folder containing the dataset
    ├── training_real/
    └── training_fake/
```

## Running the Project in Google Colab

### 1. Set Up Google Drive

1.  Create a main project folder in your Google Drive (e.g., `Deepfake_Project`).
2.  Upload the project files (`.py` files, `requirements.txt`) to this folder.
3.  Upload your dataset into a subfolder named `real_and_fake_face` inside the main project folder.

### 2. Set Up the Colab Notebook

1.  Open a new Colab notebook and change the runtime type to **T4 GPU** (`Runtime` > `Change runtime type`).
2.  Mount your Google Drive:
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```
3.  Change to your project directory. **Remember to change the path to where you saved the project.**
    ```python
    import os
    os.chdir('/content/drive/My Drive/Deepfake_Project')
    ```

### 3. Install Dependencies

Install the required Python libraries:
```python
!pip install -r requirements.txt
```

### 4. Train the Model

Before training, you may need to update the dataset paths in `train.py`. By default, they are:
*   `real = 'real_and_fake_face/training_real/'`
*   `fake = 'real_and_fake_face/training_fake/'`

Run the training script. This will save `deepfake_detection_model.h5` and some training plots in your project directory.

```python
!python train.py
```

### 5. Run the Streamlit Application with `ngrok`

Use `ngrok` to create a public URL for your Streamlit app.

1.  **Get your `ngrok` authtoken** from the [ngrok dashboard](https://dashboard.ngrok.com/get-started/your-authtoken).
2.  Run the following code in Colab, replacing `YOUR_AUTHTOKEN` with your token.

    ```python
    !pip install pyngrok
    from pyngrok import ngrok

    # Terminate any existing ngrok tunnels
    ngrok.kill()

    # Set up ngrok with your authtoken
    ngrok.set_auth_token("YOUR_AUTHTOKEN")

    # Run streamlit in background
    !nohup streamlit run app.py &

    # Create the public URL
    public_url = ngrok.connect(port='8501')
    print('Streamlit App URL:', public_url)
    ```
3. Click the `ngrok.io` link to open your application.

## How to Make a Prediction with `predict.py`

You can test a single image prediction by updating the `image_path` in `predict.py` and running the script:
```python
!python predict.py
```
This will print whether the image is "Real" or "Fake".
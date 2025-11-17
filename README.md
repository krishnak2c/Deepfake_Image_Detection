# Running the Deepfake Detection Project in Google Colab

This guide provides step-by-step instructions on how to run the deepfake detection project in a Google Colab environment.

## 1. Prerequisites

Before you begin, make sure you have the following files and folders ready:

*   **`Code` directory:** This contains the Python scripts (`app.py`, `predict.py`, `train.py`, `utils.py`).
*   **`real_and_fake_face` directory:** This is the dataset containing the training images.
*   **`coverpage.png`:** An image file used in the Streamlit application.

## 2. Upload to Google Drive

You need to upload your project files to Google Drive to access them in Colab.

1.  Create a new folder in your Google Drive, for example, `Deepfake_Detection`.
2.  Inside the `Deepfake_Detection` folder, create a `Code` folder.
3.  Upload the Python scripts (`app.py`, `predict.py`, `train.py`, `utils.py`) and the `coverpage.png` image into the `Code` folder.
4.  Upload the `real_and_fake_face` dataset directory to the `Deepfake_Detection` folder.

Your folder structure in Google Drive should look like this:

```
/My Drive/
└── Deepfake_Detection/
    ├── Code/
    │   ├── app.py
    │   ├── predict.py
    │   ├── train.py
    │   ├── utils.py
    │   └── coverpage.png
    └── real_and_fake_face/
        └── ... (dataset files)
```

## 3. Set up the Colab Notebook

1.  Open a new Colab notebook.
2.  Mount your Google Drive by running the following code in a cell:

    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

3.  Change the current directory to your project's `Code` folder. **Make sure to replace `My Drive` with your Google Drive's root folder name if it's different.**

    ```python
    import os
    os.chdir('/content/drive/My Drive/Deepfake_Detection/Code')
    ```

## 4. Install Dependencies

Install the required Python libraries by running this command in a new cell:

```python
!pip install streamlit numpy opencv-python tensorflow pandas scikit-learn matplotlib tqdm pyngrok
```

## 5. Train the Model

To train the deepfake detection model, run the `train.py` script. The script will prompt you to enter the paths for your real and fake datasets. This script has been optimized for better results and higher accuracy through enhanced data augmentation, fine-tuned model architecture, and advanced callbacks (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau). It also includes TPU compatibility for faster training if a TPU runtime is available.

This will save the trained model as `deepfake_detection_model.h5` and generate the training plots (`Figure_1.png` and `Figure_2.png`) in your `Code` directory.

```python
!python train.py
```

## 6. Run the Streamlit Application with `ngrok`

To run the Streamlit application, you'll need to use `ngrok` to create a public URL for your app.

1.  **Get your `ngrok` authtoken.**
    *   Go to the [ngrok dashboard](https://dashboard.ngrok.com/get-started/your-authtoken) and copy your authtoken.

2.  **Set your `ngrok` authtoken in the Colab notebook.** Replace `YOUR_AUTHTOKEN` with your actual token.

    ```python
    from pyngrok import ngrok
    ngrok.set_auth_token("YOUR_AUTHTOKEN")
    ```

3.  **Run the Streamlit app.** The following code will start the Streamlit app and expose it with `ngrok`.

    ```python
    # Terminate any existing ngrok tunnels
    ngrok.kill()

    # Set up the public URL
    public_url = ngrok.connect(port='8501')
    print('Streamlit App URL:', public_url)

    # Run the Streamlit app
    !streamlit run app.py --server.port 8501
    ```

## 7. Achieving the Output

After running the final cell, you will see a URL ending with `ngrok.io`. Click on this URL to open the Streamlit application in a new browser tab.

You can then upload an image, and the application will predict whether the image is "Real" or "Fake" and display the result.

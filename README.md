# Image_FFT_Analyzer
 This is mainly for the break down of images using FFT and IFT (Inverse fourier transformation) to analyze the differences between natural and ai generated images.

I have also added a demo so that it can train a model using the same method for the images.

Made as a project for the engineering mathematics project

## Installation

1. ```cd Image_FFT_Analyzer ```

2. ```pip install streamlit```

3. ```pip install -r requirements.txt```

4. ```streamlit run app.py```

## Files

1. ```app.py``` : Main streamlit and python file that runs fft and ift along with cnn for image analyzing

2. ```train_cnn.py```: File that outputs a training data and model weights for an ai model


### Important Notes
- The model training is a demo and the data can be easily overfit so highly recommend to add more features and more dataset.

- We added a folder for differences between ai and real images after analyzing 50 total images*


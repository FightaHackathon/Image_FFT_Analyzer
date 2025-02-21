import streamlit as st
import numpy as np
import cv2
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import os
import subprocess

# Dummy CNN Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))  # First conv layer activation
        x2 = F.relu(self.conv2(x1))
        x3 = F.adaptive_avg_pool2d(x2, (8, 8))
        x4 = x3.view(x3.size(0), -1)
        x5 = F.relu(self.fc1(x4))
        x6 = self.fc2(x5)
        return x6, x1  # Return both output and first layer activations

# FFT processing functions
def apply_fft(image):
    fft_channels = []
    for channel in cv2.split(image):
        fft = np.fft.fft2(channel)
        fft_shifted = np.fft.fftshift(fft)
        fft_channels.append(fft_shifted)
    return fft_channels

def filter_fft_percentage(fft_channels, percentage):
    filtered_fft = []
    for fft_data in fft_channels:
        magnitude = np.abs(fft_data)
        sorted_mag = np.sort(magnitude.flatten())[::-1]
        num_keep = int(len(sorted_mag) * percentage / 100)
        threshold = sorted_mag[num_keep - 1] if num_keep > 0 else 0
        mask = magnitude >= threshold
        filtered_fft.append(fft_data * mask)
    return filtered_fft

def inverse_fft(filtered_fft):
    reconstructed_channels = []
    for fft_data in filtered_fft:
        fft_ishift = np.fft.ifftshift(fft_data)
        img_reconstructed = np.fft.ifft2(fft_ishift).real
        img_normalized = cv2.normalize(img_reconstructed, None, 0, 255, cv2.NORM_MINMAX)
        reconstructed_channels.append(img_normalized.astype(np.uint8))
    return cv2.merge(reconstructed_channels)

# CNN Pass Visualization
def pass_to_cnn(fft_image):
    model = SimpleCNN()
    magnitude_tensor = torch.tensor(np.abs(fft_image), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        output, activations = model(magnitude_tensor)
    
    # Ensure activations have the correct shape [batch_size, channels, height, width]
    if len(activations.shape) == 3:
        activations = activations.unsqueeze(0)  # Add batch dimension if missing
    
    return activations, magnitude_tensor

# 3D plotting function
def create_3d_plot(fft_channels, downsample_factor=1):
    fig = make_subplots(
        rows=3, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}],
               [{'type': 'scene'}, {'type': 'scene'}],
               [{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=(
            'Blue - Magnitude', 'Blue - Phase',
            'Green - Magnitude', 'Green - Phase',
            'Red - Magnitude', 'Red - Phase'
        )
    )

    for i, fft_data in enumerate(fft_channels):
        fft_down = fft_data[::downsample_factor, ::downsample_factor]
        magnitude = np.abs(fft_down)
        phase = np.angle(fft_down)
        
        rows, cols = magnitude.shape
        x = np.linspace(-cols//2, cols//2, cols)
        y = np.linspace(-rows//2, rows//2, rows)
        X, Y = np.meshgrid(x, y)

        fig.add_trace(
            go.Surface(x=X, y=Y, z=magnitude, colorscale='Viridis', showscale=False),
            row=i+1, col=1
        )
        
        fig.add_trace(
            go.Surface(x=X, y=Y, z=phase, colorscale='Inferno', showscale=False),
            row=i+1, col=2
        )

    fig.update_layout(
        height=1500,
        width=1200,
        margin=dict(l=0, r=0, b=0, t=30),
        scene_camera=dict(eye=dict(x=1.5, y=1.5, z=0.5)),
        scene=dict(
            xaxis=dict(title='Frequency X'),
            yaxis=dict(title='Frequency Y'),
            zaxis=dict(title='Magnitude/Phase')
        )
    )
    return fig

# Streamlit UI
st.set_page_config(layout="wide")
st.title("Interactive Frequency Domain Analysis with CNN")

# Initialize session state
if 'fft_channels' not in st.session_state:
    st.session_state.fft_channels = None
if 'filtered_fft' not in st.session_state:
    st.session_state.filtered_fft = None
if 'reconstructed' not in st.session_state:
    st.session_state.reconstructed = None
if 'show_cnn' not in st.session_state:
    st.session_state.show_cnn = False

# Define multipage functionality
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Train Model"])

    if page == "Home":
        home_page()
    elif page == "Train Model":
        train_model_page()

def home_page():
    # Upload image
    uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        file_bytes = np.frombuffer(uploaded_file.getvalue(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image_rgb, caption="Original Image", use_column_width=True)

        # Apply FFT and store in session state
        if st.session_state.fft_channels is None:
            st.session_state.fft_channels = apply_fft(image)

        # Frequency percentage slider
        percentage = st.slider(
            "Percentage of frequencies to retain:",
            0.1, 100.0, 10.0, 0.1,
            help="Adjust the slider to select what portion of frequency components to keep."
        )

        # Apply FFT filter
        if st.button("Apply Filter"):
            st.session_state.filtered_fft = filter_fft_percentage(st.session_state.fft_channels, percentage)
            st.session_state.reconstructed = inverse_fft(st.session_state.filtered_fft)
            st.session_state.show_cnn = False  # Reset CNN visualization

        # Display reconstructed image and FFT data
        if st.session_state.reconstructed is not None:
            reconstructed_rgb = cv2.cvtColor(st.session_state.reconstructed, cv2.COLOR_BGR2RGB)
            st.image(reconstructed_rgb, caption="Reconstructed Image", use_column_width=True)

            # FFT Data Tables
            st.subheader("Frequency Data of Each Channel")
            for i, channel_name in enumerate(['Blue', 'Green', 'Red']):
                st.write(f"### {channel_name} Channel FFT Data")
                magnitude_df = pd.DataFrame(np.abs(st.session_state.filtered_fft[i]))
                phase_df = pd.DataFrame(np.angle(st.session_state.filtered_fft[i]))
                st.write("#### Magnitude Data:")
                st.dataframe(magnitude_df.head(10))
                st.write("#### Phase Data:")
                st.dataframe(phase_df.head(10))

            # 3D Visualization
            st.subheader("3D Frequency Components Visualization")
            downsample = st.slider(
                "Downsampling factor for 3D plots:",
                1, 20, 5,
                help="Controls the resolution of the 3D surface plots."
            )
            fig = create_3d_plot(st.session_state.filtered_fft, downsample)
            st.plotly_chart(fig, use_container_width=True)

            # Custom CSS to style the button
            st.markdown("""
                <style>
                    .centered-button {
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        margin-top: 20px;
                    }
                    .stButton>button {
                        padding: 20px 40px;
                        font-size: 20px;
                        background-color: #4CAF50;
                        color: white;
                        border: none;
                        border-radius: 10px;
                        cursor: pointer;
                    }
                    .stButton>button:hover {
                        background-color: #45a049;
                    }
                </style>
            """, unsafe_allow_html=True)

            # CNN Visualization Section
            with st.container():
                st.markdown('<div class="centered-button">', unsafe_allow_html=True)
                if st.button("Pass to CNN"):
                    st.session_state.show_cnn = True
                st.markdown('</div>', unsafe_allow_html=True)

            if st.session_state.show_cnn:
                st.subheader("CNN Processing Visualization")
                activations, magnitude_tensor = pass_to_cnn(st.session_state.filtered_fft[0])
                
                # Display input tensor with improved visualization
                st.write("### Input Magnitude Tensor")
                fig_input, ax_input = plt.subplots(figsize=(8, 8))
                input_img = magnitude_tensor.squeeze().numpy()
                im = ax_input.imshow(input_img, cmap='viridis')
                plt.colorbar(im, ax=ax_input)
                st.pyplot(fig_input)
                
                # Display activation maps with proper normalization
                st.write("### First Convolution Layer Activations")
                activation = activations.detach().numpy()
                
                if len(activation.shape) == 4:
                    # Create grid layout for activation maps
                    st.write("#### Activation Maps Visualization")
                    cols = 4
                    rows = 4
                    fig, axs = plt.subplots(rows, cols, figsize=(20, 20))
                    
                    for i in range(activation.shape[1]):
                        ax = axs[i//cols, i%cols]
                        act_img = activation[0, i, :, :]
                        vmin, vmax = np.percentile(act_img, [1, 99])  # Robust normalization
                        im = ax.imshow(act_img, cmap='inferno', vmin=vmin, vmax=vmax)
                        ax.set_title(f'Channel {i+1}')
                        fig.colorbar(im, ax=ax)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Display activation statistics
                    st.write("#### Activation Value Distribution")
                    flat_activations = activation.flatten()
                    fig_hist = px.histogram(
                        x=flat_activations,
                        nbins=100,
                        title="Activation Value Distribution",
                        labels={'x': 'Activation Value'}
                    )
                    st.plotly_chart(fig_hist)

                # Second Convolution Layer Visualization
                st.markdown("---")
                st.subheader("Second Convolution Layer Features")
                with torch.no_grad():
                    model = SimpleCNN()
                    _, first_conv = model(magnitude_tensor)
                    second_conv = model.conv2(first_conv).detach().numpy()
                
                if len(second_conv.shape) == 4:
                    # Display sample feature maps
                    st.write("#### Feature Maps Visualization")
                    cols = 8
                    rows = 4
                    fig2, axs2 = plt.subplots(rows, cols, figsize=(20, 10))
                    
                    for i in range(32):  # For all 32 channels
                        ax = axs2[i//cols, i%cols]
                        feature_map = second_conv[0, i, :, :]
                        vmin, vmax = np.percentile(feature_map, [1, 99])
                        im = ax.imshow(feature_map, cmap='plasma', vmin=vmin, vmax=vmax)
                        ax.set_title(f'FM {i+1}')
                        ax.axis('off')
                    
                    plt.tight_layout()
                    st.pyplot(fig2)

                # Pooling Layer Visualization
                st.markdown("---")
                st.subheader("Pooling Layer Output")
                with torch.no_grad():
                    pooled = F.adaptive_avg_pool2d(torch.tensor(second_conv), (8, 8)).numpy()

                st.write("#### Pooled Features Dimensionality Reduction")

                # Create a heatmap using seaborn
                fig_pool, ax_pool = plt.subplots(figsize=(10, 6))
                sns.heatmap(
                    pooled[0, 0],  # Use the first channel of the pooled features
                    annot=True,     # Show values in each cell
                    fmt=".2f",      # Format values to 2 decimal places
                    cmap="coolwarm",# Use a color map for better visualization
                    ax=ax_pool      # Plot on the created axis
                )
                st.pyplot(fig_pool)

                # Create a grid of pooled feature maps
                cols = 4
                rows = 2
                fig, axs = plt.subplots(rows, cols, figsize=(20, 10))

                for i in range(rows * cols):
                    ax = axs[i // cols, i % cols]
                    sns.heatmap(
                        pooled[0, i],
                        annot=True,
                        fmt=".2f",
                        cmap="coolwarm",
                        ax=ax
                    )
                    ax.set_title(f"Channel {i+1}")

                plt.tight_layout()
                st.pyplot(fig)

                # Fully Connected Layer Visualization
                st.markdown("---")
                st.subheader("Fully Connected Layer Analysis")
                with torch.no_grad():
                    model = SimpleCNN()
                    flattened = model.conv2(model.conv1(magnitude_tensor))
                    flattened = F.adaptive_avg_pool2d(flattened, (8, 8))
                    flattened = flattened.view(flattened.size(0), -1)
                    fc_output = model.fc1(flattened).detach().numpy()
                
                st.write("#### FC Layer Activation Patterns")
                fig_fc = px.imshow(
                    fc_output.T,
                    labels=dict(x="Neurons", y="Features", color="Activation"),
                    color_continuous_scale="viridis"
                )
                st.plotly_chart(fig_fc)

                # Full Pipeline Explanation
                st.markdown("""
                    ### Complete Processing Pipeline
                    <div style="
                        background-color: #f0f2f6;
                        padding: 30px;
                        border-radius: 15px;
                        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
                        font-family: 'Arial', sans-serif;
                        font-size: 16px;
                        color: #333;
                        border: 1px solid #dcdcdc;
                    ">
                    <ul style="list-style-type: none; padding-left: 0;">
                        <li><strong>1. Input Preparation:</strong> Magnitude spectrum from FFT</li>
                        <li><strong>2. Feature Extraction:</strong>
                            <ul>
                                <li>- Conv1: 16 filters (3x3)</li>
                                <li>- Conv2: 32 filters (3x3)</li>
                            </ul>
                        </li>
                        <li><strong>3. Dimensionality Reduction:</strong> Adaptive average pooling (8x8)</li>
                        <li><strong>4. Feature Transformation:</strong>
                            <ul>
                                <li>- Flattening: 32×8×8 → 2048 features</li>
                                <li>- FC1: 2048 → 128 dimensions</li>
                            </ul>
                        </li>
                        <li><strong>5. Classification:</strong> FC2: 128 → 10 classes</li>
                    </ul>
                    </div>
                """, unsafe_allow_html=True)

                # Option to filter another image
                st.markdown("---")
                st.subheader("Filter Another Image")
                if st.button("Filter Another Image"):
                    st.session_state.fft_channels = None
                    st.session_state.filtered_fft = None
                    st.session_state.reconstructed = None
                    st.session_state.show_cnn = False
                    st.experimental_rerun()

def train_model_page():
    st.title("Train Model")
    
    # Directory inputs for natural and AI images
    folderA = st.text_input("Path to Natural Images Folder", "Natural_images")
    folderB = st.text_input("Path to AI Images Folder", "AI_images")
    
    if st.button("Train Model"):
        # Check if CSV and model weights already exist
        csv_exists = os.path.exists('fft_features.csv')
        model_exists = os.path.exists('cnn_model_weights.pth')
        
        if csv_exists and model_exists:
            st.warning("CSV and model weights already exist. They will be updated.")
        
        # Show progress bar
        with st.spinner("Extracting FFT features and training model..."):
            # Run train_cnn.py with the provided directories
            command = f"python train_cnn.py {folderA} {folderB}"
            subprocess.run(command, shell=True)
        st.success("FFT extraction and model training started. Check the console for progress.")

if __name__ == "__main__":
    main()

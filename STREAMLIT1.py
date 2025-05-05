# === FINAL Streamlit App with Summary Dashboard and Tumor Metrics ===

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import numpy as np
import nibabel as nib
import torch
import matplotlib.pyplot as plt
from monai.networks.nets import UNet
import tempfile

st.set_page_config(page_title="Brain Tumor Segmentation - BraTS 2020", layout="wide")

@st.cache_resource

def load_model():
    model = UNet(spatial_dims=2, in_channels=4, out_channels=1,
                 channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2)
    model.load_state_dict(torch.load("unet_brats20_final.pth", map_location="cpu"))
    model.eval()
    return model

def preprocess_image(flair, t1, t1ce, t2):
    flair = (flair - flair.min()) / (flair.max() - flair.min() + 1e-8)
    t1 = (t1 - t1.min()) / (t1.max() - t1.min() + 1e-8)
    t1ce = (t1ce - t1ce.min()) / (t1ce.max() - t1ce.min() + 1e-8)
    t2 = (t2 - t2.min()) / (t2.max() - t2.min() + 1e-8)
    image = np.stack([flair, t1, t1ce, t2], axis=0)
    return torch.tensor(image).float().unsqueeze(0)

def predict_mask(model, image_tensor, threshold=0.5):
    with torch.no_grad():
        output = model(image_tensor)
        mask = (output > threshold).float()
    return mask.squeeze().cpu().numpy(), output.squeeze().cpu().numpy()

st.title("🧠 Brain Tumor Segmentation - BraTS 2020")

model = load_model()

tabs = st.tabs(["Upload Scans", "Single Slice Visualization", "3D Volume Overview", "📈 Results Summary"])

with tabs[0]:
    st.header("Upload MRI Scans")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        flair_file = st.file_uploader("Upload FLAIR (.nii)", type=["nii", "nii.gz"], key="flair")
    with col2:
        t1_file = st.file_uploader("Upload T1 (.nii)", type=["nii", "nii.gz"], key="t1")
    with col3:
        t1ce_file = st.file_uploader("Upload T1ce (.nii)", type=["nii", "nii.gz"], key="t1ce")
    with col4:
        t2_file = st.file_uploader("Upload T2 (.nii)", type=["nii", "nii.gz"], key="t2")

if flair_file and t1_file and t1ce_file and t2_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as tmp_flair, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as tmp_t1, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as tmp_t1ce, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as tmp_t2:

        tmp_flair.write(flair_file.read())
        tmp_t1.write(t1_file.read())
        tmp_t1ce.write(t1ce_file.read())
        tmp_t2.write(t2_file.read())

        tmp_flair.flush()
        tmp_t1.flush()
        tmp_t1ce.flush()
        tmp_t2.flush()

        flair = nib.load(tmp_flair.name).get_fdata()
        t1 = nib.load(tmp_t1.name).get_fdata()
        t1ce = nib.load(tmp_t1ce.name).get_fdata()
        t2 = nib.load(tmp_t2.name).get_fdata()

    with tabs[1]:
        st.header("Single Slice Visualization and Prediction")
        threshold = st.slider("Prediction Threshold", 0.1, 0.9, 0.5, step=0.05)
        slice_choice = st.slider("Slice Index", min_value=0, max_value=flair.shape[2]-1, value=77)
        colormap_choice = st.selectbox("Choose Colormap", ["gray", "viridis", "plasma", "inferno", "magma"])
        overlay_alpha = st.slider("Overlay Transparency", 0.1, 1.0, 0.5)

        flair_slice = flair[:, :, slice_choice]
        t1_slice = t1[:, :, slice_choice]
        t1ce_slice = t1ce[:, :, slice_choice]
        t2_slice = t2[:, :, slice_choice]

        image_tensor = preprocess_image(flair_slice, t1_slice, t1ce_slice, t2_slice)
        predicted_mask, raw_output = predict_mask(model, image_tensor, threshold)
        tumor_area = np.sum(predicted_mask)
        voxel_volume_mm3 = 1 * 1 * 1  # assuming isotropic 1mm^3 voxels
        tumor_volume_mm3 = tumor_area * voxel_volume_mm3
        tumor_volume_cm3 = tumor_volume_mm3 / 1000

        st.subheader(f"Visualizing Slice {slice_choice}")
        st.metric("Predicted Tumor Area (px)", int(tumor_area))
        st.metric("Tumor Volume (mm³)", f"{tumor_volume_mm3:.2f}")
        st.metric("Tumor Volume (cm³)", f"{tumor_volume_cm3:.2f}")

        fig, ax = plt.subplots(2, 3, figsize=(18, 10))
        ax[0, 0].imshow(flair_slice, cmap=colormap_choice)
        ax[0, 0].set_title("FLAIR Input")
        ax[0, 1].imshow(t1_slice, cmap=colormap_choice)
        ax[0, 1].set_title("T1 Input")
        ax[0, 2].imshow(t1ce_slice, cmap=colormap_choice)
        ax[0, 2].set_title("T1ce Input")
        ax[1, 0].imshow(t2_slice, cmap=colormap_choice)
        ax[1, 0].set_title("T2 Input")
        ax[1, 1].imshow(predicted_mask, cmap='Reds')
        ax[1, 1].set_title("Predicted Mask")
        ax[1, 2].imshow(flair_slice, cmap=colormap_choice)
        ax[1, 2].imshow(predicted_mask, cmap='Reds', alpha=overlay_alpha)
        ax[1, 2].set_title("Overlay")
        for a in ax.flatten():
            a.axis('off')
        st.pyplot(fig)

    with tabs[2]:
        st.header("3D Volume Overview (Middle Slices)")
        mid_slices = [flair.shape[2]//4, flair.shape[2]//2, 3*flair.shape[2]//4]

        fig2, axes = plt.subplots(1, 3, figsize=(20, 6))
        for i, mid_slice in enumerate(mid_slices):
            flair_slice = flair[:, :, mid_slice]
            image_tensor = preprocess_image(
                flair[:, :, mid_slice],
                t1[:, :, mid_slice],
                t1ce[:, :, mid_slice],
                t2[:, :, mid_slice]
            )
            pred_mask, _ = predict_mask(model, image_tensor, threshold)
            axes[i].imshow(flair_slice, cmap='gray')
            axes[i].imshow(pred_mask, cmap='Reds', alpha=0.5)
            axes[i].set_title(f"Overlay at Slice {mid_slice}")
            axes[i].axis('off')

        st.pyplot(fig2)

    with tabs[3]:
        st.header("📈 Results Summary")
        st.markdown("This panel summarizes the model’s segmentation performance on the selected slice.")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Slice Index", slice_choice)
        with col2:
            st.metric("Prediction Threshold", f"{threshold:.2f}")

        st.metric("Predicted Tumor Pixels", int(tumor_area))
        st.metric("Estimated Tumor Volume (mm³)", f"{tumor_volume_mm3:.2f}")
        st.metric("Estimated Tumor Volume (cm³)", f"{tumor_volume_cm3:.2f}")

st.markdown("---")
st.caption("Built for BraTS 2020 Final Submission | U-Net Model with Multi-tab Visualization, Metrics, and Summary Dashboard")

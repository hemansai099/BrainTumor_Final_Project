# 🧠 Brain Tumor Segmentation using U-Net and Streamlit

This project implements a deep learning-based pipeline for segmenting brain tumors from MRI scans using the BraTS 2020 dataset. A trained U-Net model is deployed through a Streamlit web app, enabling interactive slice-wise visualization, tumor volume estimation, and result interpretation.

---

## 📁 Dataset

**Source**: [BraTS 2020 MICCAI Challenge](https://www.med.upenn.edu/cbica/brats2020/)  
- Four modalities per patient: `FLAIR`, `T1`, `T1ce`, `T2`  
- Each modality is a 3D volume with shape ~240×240×155  
- Ground truth: Multiclass segmentation mask (whole tumor, tumor core, enhancing tumor)

**Training Split Used:**
- 70 patient volumes for training (~10,500 slices)
- 40 volumes for validation (~2,100 slices)

---

## 🧠 Model Architecture

- **Model**: 2D U-Net  
- **Library**: MONAI (Medical Open Network for AI)  
- **Input**: 4-channel 2D slices (`FLAIR`, `T1`, `T1ce`, `T2`)  
- **Output**: Binary tumor segmentation mask  
- **Loss**: Binary Cross-Entropy  
- **Optimizer**: Adam  
- **Epochs**: 10  
- **Validation Metric**: Dice Similarity Coefficient

---

## 🚧 Challenges Faced

- High memory usage with full BraTS dataset → Resolved by reducing volume count  
- System freezes during batch loading → Resolved by slicing and lazy loading  
- Compatibility issues with MONAI and PyTorch  
- Real-time rendering of large `.nii.gz` files in Streamlit  
- Preventing overfitting with small sample size

---

## 📊 Results

- 📉 **Training loss**: 0.95 → 0.51 (after 10 epochs)  
- 🎯 **Validation Dice Score**: **0.4452**

---

## 🖥️ Streamlit Web App

### 💡 Features:
- Upload `.nii` or `.nii.gz` files for FLAIR, T1, T1ce, and T2
- View slice-wise segmentation with threshold control and colormap options
- Compute and display:
  - Tumor area (pixels)
  - Tumor volume in mm³ and cm³
- Visualize overlays on 3 key slices (3D perspective)
- Results summary panel

### 📌 How to Run:

```bash
# Step 1: Create a virtual environment
conda create -n tumorenv python=3.8
conda activate tumorenv

# Step 2: Install dependencies
pip install nibabel monai matplotlib tqdm scikit-learn streamlit

# Step 3: Run the app
streamlit run STREAMLIT1.py
```

---

## 📂 File Structure

```
├── unet_brats20_final.pth         # Trained PyTorch model
├── STREAMLIT1.py                  # Streamlit app for inference
├── braintumor_final_project.ipynb # Training + evaluation notebook
├── README.md                      # Project documentation
```

---

## 🧪 Requirements

- Python 3.8+
- PyTorch ≥ 1.9
- MONAI ≥ 1.0
- Streamlit ≥ 1.4
- nibabel, scikit-learn, matplotlib, tqdm

---

## 📜 License

This project is for educational and academic use only.  
Please refer to the BraTS dataset license for data usage terms.

---

## 🙋‍♂️ Author

**Heman Sai Chagarlamudi**  
Final Year Project – Robotics and Autonomous Systems  
Arizona State University

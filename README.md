# 👗 AI-Powered Fashion Generator

This project is a **Text-to-Fashion Image Generation App** built with a custom **GAN model** to generate realistic fashion images from text prompts using **CLIP-guided supervision**.

It leverages the **DressCodePromptSketch dataset** and improves generation quality using:
- CLIP Loss  
- WGAN-GP  
- Adaptive Instance Normalization (AdaIN)  

---

## 🌐 Demo
![Demo](https://github.com/user-attachments/assets/ae594b4a-a199-4433-a2c0-946df71e55ca)

---

## 🚀 Features

### 🧠 AI Model
- ✅ Text-to-image generation using CLIP text embeddings  
- ✅ Residual Generator with AdaIN for text conditioning  
- ✅ Spectral Normalized Discriminator  
- ✅ CLIP loss for text-image alignment  
- ✅ Gradient Penalty with WGAN-GP  
- ✅ Visualization and checkpoint saving per epoch  
- ✅ Optimized for GPU training (Kaggle/Colab)  
- ✅ Output:
  - 64×64 generated images  
  - Upscaled to 256×256 using Real-ESRGAN  

---

### 🖼️ UI & UX (React)
- Real-time image upscaling using canvas  
- Prompt suggestions with animated loading visuals  
- Toggle between original and enhanced images  
- Download option for generated images  
- Responsive UI with custom CSS themes  

---

### 🖥️ Backend (Flask)
- REST API with `/generate` endpoint  
- Accepts JSON prompt from frontend  
- Returns generated images  
- Handles inference and PNG conversion  
- CORS enabled for local communication  

---

## 🧠 Model Architecture

### Generator
- Fully Connected + AdaIN + ConvTranspose layers  
- Input: CLIP text embedding  
- Output: 64×64 RGB image  

### Discriminator
- Spectral Normalized CNN + Fully Connected layers  
- Conditional on:
  - Image  
  - Text embedding  

### CLIP Model
- `openai/clip-vit-base-patch32` (via 🤗 Transformers)  
- Used for:
  - Text embedding  
  - CLIP loss computation  

---

## 📦 Dataset

**Name:** `Abhi5ingh/Dresscodepromptsketch`

**Contents:**
- 📝 `text` → Prompt describing the fashion item  
- 🖼️ `image` → Ground truth image  
- ✏️ `sketch` → Optional sketch of the fashion item  

---

## 🛠️ Model Setup
```bash
pip install torch torchvision datasets transformers

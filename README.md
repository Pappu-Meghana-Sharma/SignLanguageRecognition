# SignLanguageRecognition

This project is about recognizing American Sign Language (A–Z) using a deep learning model. I used transfer learning with MobileNetV2 and trained the model on image data of hand signs. The notebook includes everything—data preprocessing, model training, evaluation, and saving models in `.h5` format.

---

## Folder Structure
SignLanguageRecognition/                         \
├── SignLanguage.ipynb # Main notebook with full pipeline \
├── models/ # Folder with saved models \
│ ├── model_v1.h5   \
│ ├── model_v2.h5   \
│ └── model_v3.h5   \
├── README.md       \

---

## Dataset

- ASL Alphabet Dataset
- Each class represents one alphabet (A to Z)
- Images resized to 224x224 and normalized
- Used data augmentation for better generalization

---

## Notebook (`SignLanguage.ipynb`)

The notebook has:
- Data loading and preprocessing
- Model building and fine-tuning
- Training and evaluation
- Confusion matrix and metrics
- Saving models as `.h5`

---
## Model Variants

### Custom CNN
Built from scratch using Conv2D, MaxPooling, Dropout, and Dense layers. Designed to learn features directly from the ASL image dataset without using any pretrained weights. Suitable for quick experimentation or environments without internet access.

### Feature Extraction
Used a pretrained MobileNetV2 model with `include_top=False` to extract high-level image features. Only the final classification layers (Dense layers) were trained, while the base model was frozen. Fast training with good accuracy.

### Fine-Tuning
Started with the same MobileNetV2 backbone but **unfroze** some deeper layers of the base model to allow backpropagation. This improved accuracy by adapting the pretrained features more closely to the ASL dataset. Used lower learning rates for stability.

---

## To Do
- Add real-time webcam input (maybe using opencv)

- Deploy with Streamlit or Flask

- Compare saved model versions

----
## Author
Meghana Sharma   
GitHub: Pappu-Meghana-Sharma




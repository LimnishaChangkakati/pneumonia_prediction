# pneumonia_prediction
This project leverages Convolutional Neural Networks (CNNs) and transfer learning to automatically detect pneumonia from chest X-ray images. Built using TensorFlow and Keras, the model classifies images as either "Normal" or "Pneumonia" with high accuracy, helping support faster and more accessible preliminary diagnosis.

Pneumonia Detection from Chest X-Rays using Deep Learning:
This project aims to detect pneumonia from chest X-ray images using Convolutional Neural Networks (CNNs) and transfer learning. The model classifies images as either Normal or Pneumonia and is trained on a real-world medical dataset. Built with TensorFlow Keras and trained in Google Colab, this project demonstrates the potential of AI in healthcare diagnostics.

🩺 Pneumonia Detection from Chest X-Rays using Deep Learning

This project aims to detect pneumonia from chest X-ray images using Convolutional Neural Networks (CNNs) and transfer learning. The model classifies images as either Normal or Pneumonia and is trained on a real-world medical dataset. Built with TensorFlow Keras and trained in Google Colab, this project demonstrates the potential of AI in healthcare diagnostics.

📁 Dataset
🔗 Source: Chest X-Ray Pneumonia Dataset - Kaggle (https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
📦 Contents:
- 🖼️ Approximately 5,800 labeled X-ray images
- 🧪 Two classes: NORMAL, PNEUMONIA
- 🗂️ Organized into training, validation, and testing sets

🧠 Model Architecture
- 🧱 Backbone: ResNet50 (pre-trained on ImageNet)
- ⚙️ Framework: TensorFlow Keras
- 🧾 Classification Type: Binary (Normal vs Pneumonia)
- 🔍 Explainability: Grad-CAM for visualizing model attention

🧰 Tech Stack
- 🐍 Python
- 🔧 TensorFlow Keras
- ☁️ Google Colab
- 📊 Matplotlib and Seaborn for visualization
- 🧠 Grad-CAM for model explainability

⚙️ Installation and Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/xray-pneumonia-detector.git
cd xray-pneumonia-detector
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Upload the dataset:
📥 Download from Kaggle: Chest X-ray Pneumonia Dataset (https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
📁 Place it in a folder named `chest_xray` in the root directory

4. Run the notebook:
📓 Open `Pneumonia_Xray_Classification.ipynb` in Google Colab
▶️ Follow cell by cell to train the model and visualize predictions

📈 Evaluation Metrics
- ✅ Accuracy
- 🎯 Precision and Recall
- 📉 Confusion Matrix
- 🔥 Grad-CAM heatmaps

🏆 Results
- 📊 Achieved approximately 72% accuracy on the test set
- 🩻 Effective at distinguishing pneumonia cases from normal images
- 🫁 Visualizations confirm focus on lung areas in X-rays

🖼️ Sample Grad-CAM Output

Image: gradcam_output.png (add the correct path in your repo)

🚀 Future Improvements
- ➕ Extend to multi-class detection such as COVID-19, tuberculosis
- 🌐 Integrate into a web app using Streamlit or Gradio
- 🧪 Train on larger datasets such as NIH ChestXray14

📝 License
This project is open-source under the MIT License 📄






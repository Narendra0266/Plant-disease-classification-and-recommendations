# 🌿 Plant Disease Classification and Recommendations

**Plant Disease Classification and Recommendations** is an AI-powered solution for identifying plant diseases and providing actionable recommendations. The project includes a deep learning model, **PlantCareNet (Custom CNN)**, trained on diverse plant disease datasets. Additionally, a **mobile app** is developed to allow users to upload images, detect diseases, and receive treatment recommendations.

---

## 🚀 Features  

### 📂 High-Quality & Diverse Dataset  
PlantCareNet is trained on a **carefully curated dataset** from multiple reputable sources, ensuring comprehensive coverage of plant diseases across various species.  

- 🌾 **PlantVillage** – A widely used benchmark dataset for plant disease classification.  
- 🌿 **New Bangladeshi Crop Disease Dataset** – Region-specific images covering common crop diseases in Bangladesh.  
- 🥦 **VegNet Cauliflower Dataset** – A specialized dataset for cauliflower disease identification.  
- 🍆 **Eggplant Disease Recognition Dataset** – Focused on detecting diseases affecting eggplants.  

The dataset underwent **extensive preprocessing and augmentation** to improve model generalization, making it robust for real-world applications.  


### 🌱 Deep Learning Model: PlantCareNet  
Our project is powered by **PlantCareNet**, a custom Convolutional Neural Network (CNN) specifically designed for plant disease classification. Unlike generic models, PlantCareNet is fine-tuned for **lightweight deployment**, making it suitable for mobile and edge applications. It has been trained on a diverse set of plant disease images to ensure high accuracy and robust generalization.  

- 🔹 **Custom CNN architecture** designed for plant disease identification.  
- 🔹 **Enhanced for mobile and real-time inference** with minimal latency.  
- 🔹 **Handles multiple plant species and disease categories with high precision.**  

---

### 📱 Mobile App for Disease Detection  
The project includes an **Android-based mobile application** that allows users to detect plant diseases simply by capturing an image. The app is designed with an intuitive interface, making it accessible to farmers, researchers, and agronomists.  

- 📷 **Easy-to-use interface** – Capture or upload images of the affected plant.  
- ⚡ **Real-time disease detection** – The AI model instantly processes the images.  
- 🌿 **Actionable recommendations** – Get disease information and treatment suggestions.
- 🔍 **Ref Mode & LLM Mode** – In Ref Mode, the app provides personalized recommendations based on a reference database tailored to the detected disease, offering actionable treatment suggestions. LLM Mode utilizes GPT to provide intelligent insights, answering user queries and suggesting appropriate actions, further enhancing the user experience with AI-driven support.---

---

## 📂 Repository Contents  
This repository includes the following:

- 📜 **Kaggle Notebook**: The Jupyter notebook used for training the **PlantCareNet** deep learning model. It includes data preprocessing, model architecture, training, and evaluation. The notebook is fully documented to allow easy reproduction of results.

- 📁 **Dataset**: The dataset used for training the model, which is a combination of multiple plant disease datasets such as **PlantVillage, New Bangladeshi Crop Disease, VegNet Cauliflower**, and **Eggplant Disease Recognition**. It contains images of various plant diseases across different crops like rice, corn, tomato, and more.

- 📱 **Base APK**: The compiled **Android app (.apk file)** for plant disease identification. This version of the app is ready for installation on Android devices and allows users to capture plant images, detect diseases, and get treatment recommendations.

- 📂 **Android Studio Project**: The complete source code of the mobile application developed using **Android Studio**. It contains all the necessary files for building the app from scratch, including the integration of the disease detection model, the user interface, and backend logic.

---

## 🛠 Setup and Installation

### **Prerequisites**
- **Kaggle Account**: Required to access and run the notebook. You can sign up [here](https://www.kaggle.com/).
- **Kaggle Notebook**: The model is trained in a **Kaggle Notebook** environment, which comes with pre-installed libraries such as **TensorFlow**, **Keras**, **NumPy**, and more.
- **Android Studio** (Optional): For modifying or building the Android app from source, download and install **Android Studio** from [here](https://developer.android.com/studio).

### **Running the Model in Kaggle**
1. Open the **Kaggle Notebook** provided in the repository.
2. Upload the dataset to the **Kaggle environment** (if needed).
3. Run the cells sequentially to preprocess the data, train the **PlantCareNet** model, and evaluate the results.
4. The trained model will be saved in the notebook for future inference.

### **Mobile App Setup**
1. **Base APK**: Download the **Base APK** from the repository and install it on your Android device for plant disease detection.
2. **Android Studio Project**: Open the **Android Studio Project** folder in Android Studio to modify or build the app from source.
   - After opening the project, build and deploy the app to an Android device or emulator.


### **Clone the Repository**
```sh
git clone https://github.com/Muhaimin008/Plant-disease-classification-and-recommendations.git
cd plant-disease-classification-and-recommendations
```

## Run the PlantCareNet website locally

This repository now includes a local web app for uploading a leaf photo and viewing a PlantCareNet prediction.

1. Install the app dependencies with `python3 -m pip install -r requirements.txt`.
2. Start the app with `python3 -m streamlit run app.py`.
3. Open the local address printed in the terminal (usually `http://localhost:8501`).

The app uses `model.keras` and `class_names.json` from the repository root. It does not need to download the dataset to make predictions.

### Website workflow and recommendation modes

The site follows the app flow: upload a leaf image, select its crop, choose a recommendation mode, and view the result. Selecting the crop filters the model's 35 class scores to that crop, so the selected crop must match the photo. The image is prepared at the model's 75 × 75 input size. The result then offers:

- **Standard Advice** looks up the predicted plant and disease in `recommendations.json`. For a disease class, it displays symptoms, prevention, and care text. For a healthy class, it displays prevention tips and routine care. This 35-entry reference file was recovered from the included Android app; the material is not locally verified agricultural advice. Confirm diagnoses and treatment choices with an agricultural expert and follow local product labels and rules.
- **LLM Mode** can create cautious guidance from the predicted class and optional observations. It requires an OpenAI API key; without one, Standard Advice still works. The predicted class, score, and optional note are sent to the OpenAI API when the user requests guidance. Do not enter personal or sensitive information.
- **AI response paths** differ by predicted class: healthy classes receive prevention tips; disease classes receive a description, symptoms to inspect, prevention, and safe next steps. Related quantitative guidance must not invent treatment rates or thresholds; it should explain what to measure when crop/location details are insufficient.
- **Grad-CAM focus map** is generated only when requested. It highlights image regions that influenced the class score; it is an explanation aid, not proof of a correct diagnosis.
- **Workflow and evaluation** shows the preprocessing and classification steps plus confusion-matrix and ROC images saved from the notebook. These are static notebook outputs and are not recalculated for each upload. The notebook currently uses an 85/15 train-test split; it does not implement cross-validation or automated hyperparameter tuning.

To enable LLM Mode on Streamlit Community Cloud, open the app's **Settings → Secrets** and add:

```toml
OPENAI_API_KEY = "your-api-key"
OPENAI_MODEL = "gpt-4.1-mini"
```

Keep the key in Streamlit Secrets. Do not put it in the source code or commit it to GitHub. When a user selects **Generate tailored guidance**, the app sends the predicted class, model score, and optional note to the OpenAI API.

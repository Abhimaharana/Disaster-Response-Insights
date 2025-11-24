# 📘 Disaster Response Insights & Prediction

This project analyzes natural disaster data and builds a machine learning model to predict disaster response categories. It includes data preprocessing, exploratory analysis, model training, and a Power BI dashboard for visual insights.

---

## 📊 Project Overview

The goal of this project is to:

- Analyze large-scale natural disaster data  
- Extract insights such as magnitude, fatalities, and economic loss  
- Predict disaster response categories using Logistic Regression  
- Visualize key metrics and model performance using Power BI  

This project demonstrates skills in data science, machine learning, PySpark, and dashboard reporting.

---

## 📁 Dataset Details

**Dataset:** `natural_disasters_2024.csv`  
**Location:** `data/natural_disasters_2024.csv`

The dataset includes:

| Column | Description |
|--------|-------------|
| **Disaster_Type** | Earthquake, Flood, Hurricane, Tornado, Wildfire |
| **Magnitude** | Disaster intensity |
| **Fatalities** | Number of deaths |
| **Economic_Loss($)** | Estimated economic loss in USD |
| **Location** | Country/Region |
| **Year** | Year of occurrence |

---

## 🧠 Code Structure


├── data/
│ └── natural_disasters_2024.csv
│
├── src/
│ └── train_model.py
│
├── dashboard/
│ └── disaster_dashboard.png
│
└── README.md



---

## 🧪 Model Training (PySpark)

The model uses:

- **StringIndexer** for encoding labels  
- **VectorAssembler** for combining features  
- **Logistic Regression** for classification  
- **MulticlassEvaluation** for accuracy  

**Features Used:**


---

## 📸 Dashboard Screenshot


---

## 📈 Results

### **Model Accuracy:**  
**18.69%**

### **Prediction Distribution:**
- Class **0** → 50.91%  
- Class **1** → 35.14%  
- Class **4** → 13.95%  

### **Dashboard Metrics Include:**
- Average Magnitude  
- Total Disasters  
- Average Fatalities  
- Total Economic Loss  
- Statistical summary by disaster type  
- Prediction distribution  
- Model accuracy card  

---

## 🚀 How to Run the Project

### 1. Install PySpark

### 2. Run the training script

### 3. Open the Dashboard
Use the PNG file or the Power BI `.pbix` file if available.

---

## 👨‍💻 Author

**Abhishek Maharana**  
Computer Engineering Student | Data Science Enthusiast

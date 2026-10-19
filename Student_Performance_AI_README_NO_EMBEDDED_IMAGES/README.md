# 🎓 Student Performance Prediction & Personalized Learning Recommendation System

### 🤖 Machine Learning | 📊 Data Analysis | 💡 Personalized Learning Support

**Technology:** Python · Streamlit · Pandas · NumPy · scikit-learn · XGBoost · SHAP · Plotly · Pytest

🚀 **Live Application:** https://student-performance-ai-7l6i22jkhk5staxyhunven.streamlit.app/  
💻 **GitHub Repository:** https://github.com/Tharun-720/student-performance-ai

---

## 📖 About

An interactive Python and Streamlit machine-learning application that analyses student records, estimates final performance, predicts support tiers, explains predictions using SHAP, and provides learning-support recommendations.

### 🔗 Project Links

- 🌐 **Live Application:** https://student-performance-ai-7l6i22jkhk5staxyhunven.streamlit.app/
- 💻 **GitHub Repository:** https://github.com/Tharun-720/student-performance-ai
- 📚 **UCI Student Performance Dataset:** https://archive.ics.uci.edu/dataset/320/student+performance
- 👨‍💻 **Author's GitHub Profile:** https://github.com/Tharun-720

## ✨ Features

- 📊 Exploratory Data Analysis
- 🎯 Student Assessment & Score Predictor
- 🚦 High, Medium, and Low Support Tier Classification
- 🔍 SHAP Explainability
- 💡 Personalized Learning Recommendations
- 🏆 Regression and Classification Model Benchmarking
- 🧬 Exploratory K-Means Student Clusters
- 📂 Batch CSV Scoring

---

## 🖼️ Application Screenshots

*Screenshots are linked as text below; no images are embedded in this README. Select a link to open the corresponding screenshot in the repository.*

### 🏠 1. Dashboard Overview

The main dashboard summarises the dataset and key model metrics.

🔗 [Open Dashboard Overview screenshot](docs/images/01_dashboard_overview.png)

### 📊 2. Exploratory Data Analysis

Explore final-score distribution, recorded absences, grade progression, study-time categories, and the correlation matrix.

🔗 [Open Exploratory Data Analysis screenshot](docs/images/02_exploratory_data_analysis.png)

### 🎯 3. Student Assessment & Predictor

Enter student details and run a prediction to view the estimated final score, support tier, and approximate prediction range.

🔗 [Open Student Assessment & Predictor screenshot](docs/images/03_student_predictor.png)

### 🔍 4. Explainability — SHAP

Review global feature importance and understand which inputs contribute most to the model's predictions.

🔗 [Open SHAP Explainability screenshot](docs/images/04_shap_explainability.png)

### 💡 5. Personalized Learning Recommendations

View the predicted support tier, identified learning gaps, explanations, and suggested actions.

🔗 [Open Personalized Learning Recommendations screenshot](docs/images/05_learning_recommendations.png)

### 🏆 6. Model Benchmark

Compare regression and classification models using evaluation metrics.

🔗 [Open Model Benchmark screenshot](docs/images/06_model_benchmark.png)

### 🚦 7. Confusion Matrix & Residual Analysis

Inspect classification results through a confusion matrix and regression errors through residual plots.

🔗 [Open Confusion Matrix & Residual Analysis screenshot](docs/images/07_confusion_matrix_and_residuals.png)

### 🧬 8. Student Clusters

Explore behavioural groupings using K-Means and a two-dimensional PCA visualisation. These clusters are exploratory.

🔗 [Open Student Clusters screenshot](docs/images/08_student_clusters.png)

### 📂 9. Batch Scoring

Upload a CSV file to score multiple student records. The application also provides a sample CSV.

🔗 [Open Batch Scoring screenshot](docs/images/09_batch_scoring.png)

---

## 🛠️ Technology Stack

- 🐍 **Python** — application and machine-learning workflows
- 🎨 **Streamlit** — interactive web interface
- 🧮 **Pandas and NumPy** — data handling and numerical operations
- 🤖 **scikit-learn and XGBoost** — model training and evaluation
- 🔍 **SHAP** — prediction explainability
- 📊 **Plotly and Matplotlib** — charts and visualisations
- 🧪 **Pytest** — automated tests
- 🐙 **Git and GitHub** — version control and project hosting

## ⚙️ How It Works

1. 📥 Load and validate the student dataset.
2. 🧹 Prepare features for analysis and modelling.
3. ✂️ Split records using student-aware grouping.
4. 🤖 Train and evaluate regression and classification models.
5. 🔍 Explain predictions using SHAP.
6. 💡 Generate learning-support recommendations.
7. 🖥️ Display the results in the Streamlit dashboard.

## 📁 Project Structure

```text
student-performance-ai/
├── README.md
└── docs/
    └── images/
        ├── 01_dashboard_overview.png
        ├── 02_exploratory_data_analysis.png
        ├── 03_student_predictor.png
        ├── 04_shap_explainability.png
        ├── 05_learning_recommendations.png
        ├── 06_model_benchmark.png
        ├── 07_confusion_matrix_and_residuals.png
        ├── 08_student_clusters.png
        └── 09_batch_scoring.png
```

## 🎯 Project Objectives

- Analyse student data to identify patterns associated with academic performance.
- Estimate student outcomes using machine-learning models.
- Explain model predictions in an understandable way.
- Offer personalized learning-support suggestions.
- Explore student groups and support batch scoring.

## 🚀 Run Locally

1. Clone the repository:

   ```bash
   git clone https://github.com/Tharun-720/student-performance-ai.git
   cd student-performance-ai
   ```

2. Create and activate a virtual environment (recommended).

3. Install dependencies using the project's dependency file:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the Streamlit app using the main application file in the repository:

   ```bash
   streamlit run app.py
   ```

*If your local project uses a different Streamlit entry-point filename, use that filename in the final command.*

## 🔮 Future Enhancements

- Add more student datasets for broader evaluation.
- Improve personalized recommendations with educator feedback.
- Add progress tracking across multiple assessments.
- Improve deployment, accessibility, and monitoring.

## 👨‍💻 Author

**Tharun Reddy**

- GitHub: https://github.com/Tharun-720
- Project Repository: https://github.com/Tharun-720/student-performance-ai

## 🙏 Acknowledgements

- UCI Machine Learning Repository for the Student Performance dataset.
- The open-source Python machine-learning and visualisation ecosystem.

# 🎓 Student Performance AI

An AI-powered student performance analysis and decision-support system that predicts students' final scores, identifies students who may need additional academic support, and provides explainable, personalized learning recommendations.

## 🚀 Live Demo

- **Live Application:** https://student-performance-ai-7l6i22jkhk5staxyhunven.streamlit.app/
- **GitHub Repository:** https://github.com/Tharun-720/student-performance-ai

## 📌 Project Overview

Student Performance AI is a machine learning-based web application designed to help educators understand student performance and identify potential academic difficulties early.

The system analyzes student data, predicts final academic scores, assigns support tiers, explains important prediction factors, and recommends suitable learning strategies. It also supports individual student analysis and batch predictions using CSV files.

## ✨ Key Features

- **Final Score Prediction:** Predicts student final grades using machine learning models.
- **Academic Risk Analysis:** Identifies students who may require additional academic support.
- **Explainable AI:** Uses SHAP to explain the factors influencing model predictions.
- **Personalized Recommendations:** Generates rule-based learning recommendations based on student performance.
- **Student Clustering:** Uses K-Means clustering to identify groups of students with similar characteristics.
- **Single Student Analysis:** Analyze an individual student's academic profile.
- **Batch CSV Prediction:** Upload CSV files to analyze multiple students.
- **Interactive Dashboard:** Provides an eight-page Streamlit dashboard for exploring predictions, insights, and results.
- **Model Evaluation:** Evaluates model performance using suitable machine learning metrics.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web application |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computation |
| Scikit-learn | Machine learning and clustering |
| XGBoost | Gradient boosting model |
| SHAP | Model explainability |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Pytest | Automated testing |

## 📊 Dataset

The project uses the **UCI Student Performance dataset**, containing 1,044 student records.

The dataset supports analysis of student demographics, study habits, academic history, and other available attributes relevant to academic performance.

Dataset source: [UCI Machine Learning Repository — Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance)

## 🏗️ System Workflow

1. Load and preprocess the student dataset.
2. Perform exploratory data analysis and prepare features.
3. Train and evaluate machine learning models.
4. Predict students' final scores.
5. Assign academic support categories based on the implemented rules.
6. Generate SHAP explanations for model predictions.
7. Apply K-Means clustering to identify student groups.
8. Display insights and recommendations through the Streamlit dashboard.

## 📂 Project Structure

```text
student-performance-ai/
├── app/
│   └── main.py
├── src/
│   └── run_pipeline.py
├── tests/
├── data/
├── models/
├── requirements.txt
├── README.md
└── ...
```

*Note: This is a simplified overview. The exact folders and files depend on the final project structure in your repository.*

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.10 or a compatible version supported by the project dependencies
- Git
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/Tharun-720/student-performance-ai.git
```

### 2. Navigate to the Project Directory

```bash
cd student-performance-ai
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Machine Learning Pipeline

```bash
python -m src.run_pipeline
```

### 6. Launch the Streamlit Application

```bash
streamlit run app/main.py
```

Open the local URL displayed in your terminal to access the dashboard.

## 🧪 Testing

Run the automated test suite using:

```bash
python -m pytest -q
```

The latest reported test result was **88 passed, 0 failed**.

## 🎯 Applications

- Early identification of students who may need academic support
- Data-driven academic performance analysis
- Personalized learning strategy recommendations
- Student grouping and academic pattern discovery
- Batch analysis of student records
- Educational decision support for teachers and institutions

## 🔮 Future Enhancements

- Integrate additional educational datasets.
- Add teacher and student authentication.
- Introduce longitudinal performance tracking.
- Improve recommendation personalization.
- Add downloadable PDF performance reports.
- Explore deployment and integration with institutional learning management systems.

## 👨‍💻 Author

**Tharun Kumar**

B.Tech — Computer Science and Engineering (Data Science)

GitHub: [Tharun-720](https://github.com/Tharun-720)

LinkedIn: [Connect with me](https://www.linkedin.com/in/tharunreddy05/)

## 📄 License

This project is intended for academic and educational purposes. Add a suitable open-source license if you plan to distribute the project for reuse.

---

⭐ If you find this project useful, consider starring the repository on GitHub.

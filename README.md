# 🏦 Bank Marketing MLOps Project

This project focuses on building a machine learning model to predict whether a client will subscribe to a bank term deposit. It is designed using full MLOps best practices including **experiment tracking (MLflow)**, **data/model versioning (DVC)**, and a **Streamlit web application** for prediction.

## 🔧 Tech Stack

- Python (Scikit-learn, Pandas, etc.)
- MLflow (experiment tracking)
- DVC (data & model versioning)
- Streamlit (web app)
- Git & GitHub
- VSCode

---

## 📊 Problem Statement

The goal is to predict whether a client will subscribe to a term deposit based on features such as age, job, balance, contact duration, and past campaign performance.

---

## 📁 Project Structure

bank_mlops_project/
│
├── data/ # Raw data files
│ └── bank.csv
│
├── models/ # Trained models
│ └── best_model.pkl
│
├── src/ # Source code
│ └── train.py # ML pipeline with MLflow tracking
│
├── app/ # Streamlit UI
│ └── app.py
│
├── .dvc/ # DVC config and cache
│
├── dvc.yaml # DVC pipeline definition
├── MLproject # MLflow project file
├── requirements.txt # All dependencies
├── README.md # Project documentation
└── ...

--

## ⚙️ How to Run This Project Locally

### 1. Clone the repo

```bash
git clone https://github.com/melba04george/bank_mlops_project.git
cd bank_mlops_project
2. Create & activate virtual environment
bash
Copy
Edit
python -m venv bankmarketing
bankmarketing\Scripts\activate  # On Windows
3. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
4. Run model training with MLflow
bash
Copy
Edit
python src/train.py
You can view the experiment results:

bash
Copy
Edit
mlflow ui
Then go to: http://127.0.0.1:5000

5. Track data and model with DVC
bash
Copy
Edit
dvc init
dvc add data/bank.csv
dvc add models/best_model.pkl
git add data/.gitignore models/.gitignore *.dvc dvc.yaml .dvc .gitignore
git commit -m "Track data and model with DVC"
6. Launch the Streamlit App
bash
Copy
Edit
cd app
streamlit run app.py
📸 Demo Screenshots
Home Page	Prediction

🔍 Model Performance
Accuracy: 90%

F1 Score (Positive Class): ~51%

Best Model: BaggingClassifier with hyperparameter tuning

🚀 Deployment (Coming Soon)
You can deploy this app using:

Streamlit Cloud (Free)

Hugging Face Spaces

AWS EC2 / S3 / Lambda (optional)

🧠 Next Enhancements
✅ Add SHAP explainability for model transparency

✅ CI/CD integration using GitHub Actions

✅ Deploy app for public access

📌 Author
Melba George
LinkedIn
GitHub

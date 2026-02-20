Support Ticket Intelligence System

A hybrid Machine Learning system that automatically classifies customer support tickets, assigns priority levels, and presents results through a professional dashboard interface.

This project simulates how modern SaaS companies automate ticket triage to improve response speed, reduce backlog, and support operational decision-making.

---

🚀 Features

- Automatic ticket category classification
- Priority assignment (High / Medium / Low)
- Hybrid ML pipeline:
  - TF-IDF statistical model
  - Semantic transformer model
- Model validation comparison
- Ticket ID generation
- Professional Streamlit dashboard
- Real-time ticket analysis workflow

---

🧠 How It Works

1. Ticket text is cleaned and preprocessed
2. TF-IDF model performs statistical classification
3. Semantic model understands contextual meaning
4. Results are compared for validation
5. Priority is assigned automatically
6. Dashboard displays structured decision output

This mirrors real-world support automation pipelines used in SaaS environments.

---

📂 Project Structure

```
support-ticket-ml/
│
├── app.py                  → Streamlit dashboard
├── src/
│   ├── train_model.py      → Model training pipeline
│   ├── predict.py          → Ticket prediction logic
│   └── preprocessing.py    → Text cleaning utilities
│
├── model/                  → Generated trained models (created locally)
├── requirements.txt        → Dependencies
└── README.md
```

> Note: Model files are generated locally and excluded from GitHub to keep the repository lightweight.

---

📊 Dataset Reference

This project uses a ticket classification dataset for training:

**IT Service Ticket Classification Dataset (Kaggle)**  
Large dataset containing ticket text and topic labels for multi-class classification.

🔗 https://www.kaggle.com/datasets/adisongoh/it-service-ticket-classification-dataset

Dataset is referenced for educational and project purposes.

---

⚙ Installation

Clone the repository:

```
git clone <your-repo-url>
cd support-ticket-ml
```

Install dependencies:

```
pip install -r requirements.txt
```

---

🏋 Train the Models

After downloading the dataset and placing it in the project directory:

```
python src/train_model.py
```

This generates trained models inside:

```
model/
```

---

▶ Run the Dashboard

```
streamlit run app.py
```

You can now enter support tickets and view automated classification results.

---

🎯 Example Use Cases

- IT helpdesk ticket triage
- SaaS support automation
- Internal service routing
- ML-driven decision assistance

---

🏢 Real-World Relevance

This project demonstrates:

- NLP-based ticket categorization
- Hybrid model validation
- Automated priority routing
- Support operations optimization

Similar systems are used in enterprise platforms like Zendesk, ServiceNow, and Freshdesk.

---

👨‍💻 Tech Stack

- Python
- Streamlit
- Scikit-learn
- Sentence Transformers
- NLP preprocessing tools

---

📌 Future Improvements

- Confidence scoring visualization
- Ticket history tracking
- Batch ticket processing
- Analytics dashboard
- Workflow routing simulation

---

📄 License

Educational / project use.

👤 Author

Rathod Parth Ashokbhai

Machine Learning Intern – Future Interns (2026)

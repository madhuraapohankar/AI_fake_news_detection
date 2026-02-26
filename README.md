🧠 AI Fake News Detection System

An AI-powered web application that detects whether a news article is Real or Fake using Machine Learning.

Built using Flask, Scikit-learn, and a clean SaaS-style UI.

🚀 Features

🔍 Detect fake or real news instantly

📤 Upload PDF / TXT files for analysis

📰 Live news detection interface

📊 Confidence score with visual indicator

📜 Prediction history tracking

🔐 Secure authentication (Login / Google OAuth)

🔑 Password reset via email

🎨 Light/Dark mode UI

👨‍💼 Admin dashboard support

🛠️ Tech Stack

Backend: Flask (Python)

Machine Learning: Scikit-learn

Frontend: HTML, CSS, Bootstrap

Database: SQLite

Authentication: Session-based + Google OAuth

Email Service: Flask-Mail

Environment Management: python-dotenv

📂 Project Structure
AI_fake_news_detection/
│
├── app.py
├── model/
│   ├── model.pkl
│   └── vectorizer.pkl
│
├── templates/
│   ├── index.html
│   ├── live_news.html
│   ├── upload.html
│   ├── history.html
│   └── ...
│
├── static/
│   ├── css/
│   └── script.js
│
├── database/
│
├── .gitignore
├── .env.example
└── README.md
⚙️ Installation & Setup
1️⃣ Clone Repository
git clone https://github.com/madhuraapohankar/AI_fake_news_detection.git
cd AI_fake_news_detection
2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate   # Windows
3️⃣ Install Dependencies
pip install -r requirements.txt

If requirements.txt is not available yet:

pip install flask scikit-learn numpy flask-mail python-dotenv authlib PyPDF2
pip freeze > requirements.txt
4️⃣ Setup Environment Variables

Create a .env file in the root directory:

SECRET_KEY=your_secret_key
MAIL_USERNAME=your_email@gmail.com
MAIL_PASSWORD=your_app_password
CLIENT_ID=your_google_client_id
CLIENT_SECRET=your_google_client_secret

⚠ .env is ignored via .gitignore for security.

5️⃣ Run Application
python app.py

Open in browser:

http://127.0.0.1:5000
🧪 How It Works

User inputs news text or uploads file.

Text is cleaned and preprocessed.

Vectorizer converts text into numerical features.

Trained ML model predicts:

🟢 Real News

🔴 Fake News

Confidence score is displayed.

Result optionally saved in history.

🔐 Security Practices

.env file excluded from repository

Database not pushed to GitHub

Uploaded files ignored

Session protection enabled

Secure password reset with token expiry

👨‍💻 Contributors

Archita

Madhura

Pranay

📌 Future Improvements

Improve model accuracy

Add news source credibility scoring

Integrate real-time news APIs

Deploy to cloud (Render / Azure / AWS)

Add analytics dashboard

📄 License

This project is for educational and academic purposes.

⭐ If you found this project useful, consider giving it a star!
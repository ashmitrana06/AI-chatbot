🚀 AI-Powered Chatbot | Gemini AI API + Flask + TailwindCSS

An AI-powered chatbot that combines FAQ matching with real-time AI responses from the Gemini AI API.
Built with Flask, Sentence Transformers, and TailwindCSS for a sleek, responsive interface.

✨ Features

✅ Semantic FAQ Matching – Finds answers instantly using Sentence Transformers.

✅ Real-Time AI Responses – Falls back to Gemini AI API when no FAQ match exists.

✅ Persistent Chat History – Stored in SQLite for future reference.

✅ Modern, Responsive UI – Built with TailwindCSS for mobile & desktop.

✅ Error Handling & API Retry – Backoff and retry mechanism for API calls.

🛠️ Tech Stack

  Backend: Python, Flask, SQLite, Sentence Transformers, Gemini AI API
  
  Frontend: HTML, TailwindCSS, JavaScript
  
  Other: NLTK, Requests, Backoff for API retries

📦 Installation

1️⃣ Clone the repository

    git clone https://github.com/yourusername/ai-chatbot-gemini.git
  
    cd ai-chatbot-gemini

2️⃣ Install dependencies

    pip install -r requirements.txt

3️⃣ Set your Gemini AI API key

    export GEMINI_API_KEY="your_api_key_here"
  
  (On Windows use set instead of export)

4️⃣ Run the app

    python app.py

5️⃣ Open in your browser

    http://127.0.0.1:5000

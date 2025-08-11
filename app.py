import os
import sqlite3
import json
import time
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util
import nltk
import requests # For making HTTP requests to the Gemini API
import backoff # For exponential backoff

# Ensure NLTK punkt tokenizer is available for potential future use (or just good practice)
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

app = Flask(__name__)
CORS(app) # Enable CORS for frontend communication

# --- Configuration ---
DATABASE_FILE = 'chat_logs.db'
FAQ_DATA = [
    {"question": "What are your operating hours?", "answer": "Our operating hours are Monday to Friday, 9 AM to 5 PM."},
    {"question": "How can I contact customer support?", "answer": "You can contact our customer support via email at support@example.com or call us at 1-800-555-0123."},
    {"question": "Where can I find pricing information?", "answer": "Detailed pricing information is available on our 'Pricing' page on the website."},
    {"question": "How do I reset my password?", "answer": "To reset your password, please go to the login page and click on 'Forgot Password'."},
    {"question": "Do you offer refunds?", "answer": "Our refund policy allows for full refunds within 30 days of purchase. Please refer to our Terms and Conditions for more details."},
    {"question": "What services do you provide?", "answer": "We provide a range of services including web development, mobile app development, and cloud solutions."},
    {"question": "How do I create an account?", "answer": "To create an account, click on the 'Sign Up' button at the top right corner of the page and follow the instructions."},
    {"question": "Is there a free trial available?", "answer": "Yes, we offer a 14-day free trial for all our premium services. No credit card required to start."},
    {"question": "What payment methods do you accept?", "answer": "We accept major credit cards (Visa, MasterCard, American Express) and PayPal."},
    {"question": "How long does shipping take?", "answer": "Standard shipping usually takes 5-7 business days. Expedited options are available at checkout."}
]
MODEL_NAME = 'all-MiniLM-L6-v2' # A good balance of size and performance
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key="
API_KEY = "" # Leave this empty. The environment will provide it if running in Canvas.

# --- Global Variables ---
model = None
faq_embeddings = None

# --- Database Initialization ---
def init_db():
    """Initializes the SQLite database for chat logs."""
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL, -- 'user' or 'bot'
                message TEXT NOT NULL,
                source TEXT, -- 'faq' or 'gemini'
                matched_faq_question TEXT, -- For bot responses, the matched FAQ
                confidence REAL -- Confidence score of the match for FAQ
            )
        ''')
        conn.commit()
        print("Database initialized successfully.")
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        if conn:
            conn.close()

# --- NLP Model Loading ---
def load_nlp_model():
    """Loads the sentence-transformers model and pre-computes FAQ embeddings."""
    global model, faq_embeddings
    print(f"Loading NLP model: {MODEL_NAME}...")
    try:
        model = SentenceTransformer(MODEL_NAME)
        print("Model loaded successfully.")
        # Pre-compute embeddings for FAQ questions
        faq_questions = [item["question"] for item in FAQ_DATA]
        faq_embeddings = model.encode(faq_questions, convert_to_tensor=True)
        print("FAQ embeddings pre-computed.")
    except Exception as e:
        print(f"Error loading NLP model or computing embeddings: {e}")
        model = None
        faq_embeddings = None

# --- Chatbot Logic ---
def find_best_faq(user_query: str):
    """
    Finds the best matching FAQ answer for a given user query using cosine similarity.
    Returns (answer, matched_question, confidence) or (None, None, 0.0) if no good match.
    """
    if model is None or faq_embeddings is None:
        print("NLP model not loaded. Cannot find FAQ.")
        return None, None, 0.0

    user_query_embedding = model.encode(user_query, convert_to_tensor=True)

    # Compute cosine similarity between the user query and all FAQ questions
    cosine_scores = util.cos_sim(user_query_embedding, faq_embeddings)[0]

    # Find the best match
    best_score = 0.0
    best_match_index = -1

    for i, score in enumerate(cosine_scores):
        if score > best_score:
            best_score = score
            best_match_index = i

    # Define a confidence threshold for FAQ
    FAQ_CONFIDENCE_THRESHOLD = 0.70 # Increased threshold for a more precise FAQ match
    if best_score >= FAQ_CONFIDENCE_THRESHOLD:
        matched_faq = FAQ_DATA[best_match_index]
        return matched_faq["answer"], matched_faq["question"], best_score
    else:
        # No sufficiently confident match for FAQ
        return None, None, best_score # Indicate no confident FAQ match

@backoff.on_exception(backoff.expo,
                      requests.exceptions.RequestException,
                      max_tries=5,
                      factor=2) # Retry with exponential backoff for network errors
def call_gemini_api(prompt: str) -> str:
    """Calls the Gemini API to generate a response."""
    print("Calling Gemini API...")
    url = f"{GEMINI_API_URL}{API_KEY}" # API_KEY will be provided by the environment
    headers = {'Content-Type': 'application/json'}
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}]
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        result = response.json()
        if result.get("candidates") and result["candidates"][0].get("content") and result["candidates"][0]["content"].get("parts"):
            text = result["candidates"][0]["content"]["parts"][0]["text"]
            return text
        else:
            print(f"Unexpected Gemini API response structure: {json.dumps(result)}")
            return "I couldn't generate a response from the AI. Please try again."
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error calling Gemini API: {http_err} - {response.text}")
        return "There was an issue connecting to the AI service. Please try again later."
    except requests.exceptions.ConnectionError as conn_err:
        print(f"Connection error calling Gemini API: {conn_err}")
        return "A network error occurred while trying to reach the AI. Please check your internet connection."
    except requests.exceptions.Timeout as timeout_err:
        print(f"Timeout error calling Gemini API: {timeout_err}")
        return "The AI service took too long to respond. Please try again."
    except requests.exceptions.RequestException as req_err:
        print(f"An unexpected error occurred calling Gemini API: {req_err}")
        return "An unexpected error occurred with the AI service. Please try again."
    except json.JSONDecodeError as json_err:
        print(f"JSON decode error from Gemini API response: {json_err} - Response: {response.text}")
        return "Received an unreadable response from the AI. Please try again."
    except Exception as e:
        print(f"An unknown error occurred in call_gemini_api: {e}")
        return "An internal error prevented me from generating a response."


def log_interaction(user_id: str, role: str, message: str, source: str = None, matched_faq_question: str = None, confidence: float = None):
    """Logs a chat interaction to the database."""
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())
        cursor.execute(
            "INSERT INTO chat_history (timestamp, user_id, role, message, source, matched_faq_question, confidence) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (timestamp, user_id, role, message, source, matched_faq_question, confidence)
        )
        conn.commit()
    except sqlite3.Error as e:
        print(f"Error logging interaction: {e}")
    finally:
        if conn:
            conn.close()

# --- Flask Routes ---
@app.route('/chat', methods=['POST'])
def chat():
    """Handles incoming chat messages from the frontend."""
    data = request.json
    user_message = data.get('message')
    user_id = data.get('userId', 'anonymous_user') # Get user ID from frontend, default to anonymous

    if not user_message:
        return jsonify({"response": "Please provide a message.", "role": "bot"}), 400

    # Log user message
    log_interaction(user_id, 'user', user_message)

    # First, try to find a best FAQ match
    faq_answer, matched_question, faq_confidence = find_best_faq(user_message)

    bot_response = ""
    response_source = "gemini" # Default source is Gemini
    if faq_answer:
        # If a confident FAQ match is found, use it
        bot_response = faq_answer
        response_source = "faq"
    else:
        # If no confident FAQ match, fallback to Gemini API
        print(f"No confident FAQ match (score: {faq_confidence:.2f}). Falling back to Gemini.")
        bot_response = call_gemini_api(user_message)
        # Gemini doesn't have a direct 'matched_question' or 'confidence' in the same way as FAQ
        matched_question = None
        faq_confidence = None

    # Log bot response
    log_interaction(user_id, 'bot', bot_response, response_source, matched_question, faq_confidence)

    return jsonify({
        "response": bot_response,
        "role": "bot",
        "source": response_source, # Indicate if response came from FAQ or Gemini
        "matched_question": matched_question,
        "confidence": faq_confidence
    })

@app.route('/history', methods=['GET'])
def get_history():
    """Retrieves the full chat history from the database."""
    user_id = request.args.get('userId', 'anonymous_user') # Filter history by user ID
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        # Retrieve all columns including 'source'
        cursor.execute("SELECT timestamp, role, message, source, matched_faq_question, confidence FROM chat_history WHERE user_id = ? ORDER BY timestamp ASC", (user_id,))
        history = cursor.fetchall()
        # Convert to a list of dictionaries for easier consumption by frontend
        history_list = []
        for item in history:
            history_list.append({
                "timestamp": item[0],
                "role": item[1],
                "message": item[2],
                "source": item[3], # Include source
                "matched_faq_question": item[4],
                "confidence": item[5]
            })
        return jsonify(history_list)
    except sqlite3.Error as e:
        print(f"Error fetching history: {e}")
        return jsonify([]), 500
    finally:
        if conn:
            conn.close()

@app.route('/')
def index():
    """Serves the main HTML page."""
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    """Serves other static files (like CSS/JS)."""
    return send_from_directory('.', path)

# --- Application Startup ---
if __name__ == '__main__':
    # Initialize database
    init_db()
    # Load NLP model (This will download the model if not present locally)
    load_nlp_model()
    # Run the Flask app
    # Use 0.0.0.0 to make it accessible externally if running in a container/VM
    app.run(debug=True, host='0.0.0.0', port=5000)

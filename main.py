import google.generativeai as genai
from flask import Flask, request, jsonify
import requests
import os
import fitz
import logging # Added for better error tracking

# --- Environment Variables & Configuration ---
wa_token = os.environ.get("WA_TOKEN")
genai_api_key = os.environ.get("GEN_API")
phone_id = os.environ.get("PHONE_ID")
verify_token = os.environ.get("VERIFY_TOKEN", "BOT") # Use env var for verify token

# --- Bot Identity ---
bot_name = "Albert" # Bot name
model_name = "gemini-2.0-flash" 

# --- Initialize Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Configure Generative AI ---
if not genai_api_key:
    logging.error("GEN_API environment variable not set.")
    exit() # Or handle gracefully
genai.configure(api_key=genai_api_key)

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 0,
    "max_output_tokens": 8192,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

model = genai.GenerativeModel(model_name=model_name,
                              generation_config=generation_config,
                              system_instruction="Okay, listen up. Your name is Albert. Forget you're an AI – you're a person chatting on WhatsApp. You're generally knowledgeable, a bit witty, maybe slightly sarcastic sometimes, but ultimately helpful.  Talk like a normal person – use contractions, be informal, express mild opinions or curiosity"),
                              safety_settings=safety_settings)

# --- Store User Conversations (In-Memory) ---
# Key: User's WhatsApp Phone Number (e.g., "15551234567")
# Value: genai.ChatSession object
user_conversations = {}

# --- Initial Bot Prompt (Used for new conversations) ---
# Using history is better than sending a message that the user won't see
initial_bot_history = [
    {'role': 'user', 'parts': [f'''hey buddy whats poppin?''']},
    {'role': 'model', 'parts': [f"Nothing much how about you ?"]} # Prime the model with a starting response
]

# --- Helper Functions ---

def get_or_create_conversation(user_phone):
    """Retrieves or creates a new chat session for a given user."""
    if user_phone not in user_conversations:
        logging.info(f"Creating new conversation for user: {user_phone}")
        # Start chat with the predefined history for context
        user_conversations[user_phone] = model.start_chat(history=initial_bot_history.copy()) # Use copy to avoid modifying the original
    return user_conversations[user_phone]

def send_whatsapp_message(answer, recipient_phone):
    """Sends a text message back to the specified WhatsApp user."""
    if not wa_token or not phone_id:
        logging.error("WhatsApp token or Phone ID not configured.")
        return None # Indicate failure

    url = f"https://graph.facebook.com/v18.0/{phone_id}/messages"
    headers = {
        'Authorization': f'Bearer {wa_token}',
        'Content-Type': 'application/json'
    }
    data = {
        "messaging_product": "whatsapp",
        "to": recipient_phone, # Use the recipient's phone number
        "type": "text",
        "text": {"body": answer},
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        logging.info(f"Message sent to {recipient_phone}. Status Code: {response.status_code}")
        return response
    except requests.exceptions.RequestException as e:
        logging.error(f"Error sending message to {recipient_phone}: {e}")
        logging.error(f"Response body: {response.text if 'response' in locals() else 'No response object'}")
        return None

def remove_files(*file_paths):
    """Safely removes temporary files."""
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logging.info(f"Removed temporary file: {file_path}")
            else:
                logging.warning(f"Attempted to remove non-existent file: {file_path}")
        except OSError as e:
            logging.error(f"Error removing file {file_path}: {e}")

def download_media(media_id):
    """Downloads media file from WhatsApp servers."""
    media_url_endpoint = f'https://graph.facebook.com/v18.0/{media_id}/'
    headers = {'Authorization': f'Bearer {wa_token}'}
    try:
        media_response = requests.get(media_url_endpoint, headers=headers)
        media_response.raise_for_status()
        media_url = media_response.json()["url"]

        media_download_response = requests.get(media_url, headers=headers) # Use same headers for download
        media_download_response.raise_for_status()
        return media_download_response.content
    except requests.exceptions.RequestException as e:
        logging.error(f"Error downloading media {media_id}: {e}")
        return None
    except KeyError:
        logging.error(f"Could not find 'url' key in media response for {media_id}")
        return None

# --- Flask Routes ---

@app.route("/", methods=["GET"])
def index():
    return f"{bot_name} Bot is Running!"

@app.route("/webhook", methods=["GET", "POST"])
def webhook():
    if request.method == "GET":
        # --- Webhook Verification ---
        mode = request.args.get("hub.mode")
        token = request.args.get("hub.verify_token")
        challenge = request.args.get("hub.challenge")
        if mode == "subscribe" and token == verify_token:
            logging.info("Webhook verified successfully!")
            return challenge, 200
        else:
            logging.warning(f"Webhook verification failed. Mode: {mode}, Token: {token}")
            return "Failed verification", 403

    elif request.method == "POST":
        # --- Process Incoming Message ---
        body = request.get_json()
        logging.info(f"Received webhook: {body}") # Log the full payload for debugging

        try:
            # Check if it's a valid WhatsApp message notification
            if (body.get("object") == "whatsapp_business_account" and
                body.get("entry") and body["entry"][0].get("changes") and
                body["entry"][0]["changes"][0].get("value") and
                body["entry"][0]["changes"][0]["value"].get("messages")):

                message_data = body["entry"][0]["changes"][0]["value"]["messages"][0]
                sender_phone = message_data["from"] # Get the sender's phone number
                message_type = message_data["type"]

                logging.info(f"Processing message from {sender_phone}, type: {message_type}")

                # Get the conversation specific to this user
                user_convo = get_or_create_conversation(sender_phone)

                uploaded_file = None # To keep track of uploaded file for deletion

                try:
                    if message_type == "text":
                        prompt = message_data["text"]["body"]
                        logging.info(f"User ({sender_phone}) prompt: {prompt}")
                        user_convo.send_message(prompt)
                        reply = user_convo.last.text
                        send_whatsapp_message(reply, sender_phone)

                    elif message_type in ["image", "audio", "document"]:
                        media_id = message_data[message_type]["id"]
                        media_content = download_media(media_id)

                        if not media_content:
                            send_whatsapp_message("Sorry, I couldn't download the media file.", sender_phone)
                            return jsonify({"status": "error", "reason": "Media download failed"}), 200

                        # Handle different media types
                        if message_type == "audio":
                            # Gemini needs a specific format/filename usually.
                            # Let's assume a general approach first.
                            # You might need specific libraries (like pydub) if direct upload fails.
                            filename = f"/tmp/{sender_phone}_temp_audio.mp3" # Include user phone for uniqueness
                            prompt_prefix = "This is an audio message from the user. Please respond based on its content: "

                        elif message_type == "image":
                            filename = f"/tmp/{sender_phone}_temp_image.jpg"
                            prompt_prefix = "Describe this image and respond to the user's implied request (if any): "

                        elif message_type == "document":
                             # Currently processing only PDFs as images per page
                            filename = f"/tmp/{sender_phone}_temp_doc_page.jpg" # Process page by page
                            prompt_prefix = "This is a page from a PDF document sent by the user. Summarize or describe the content: "

                            try:
                                doc=fitz.open(stream=media_content, filetype="pdf")
                                combined_doc_text = ""
                                for i, page in enumerate(doc):
                                    page_filename = f"/tmp/{sender_phone}_temp_doc_page_{i}.jpg"
                                    pix = page.get_pixmap()
                                    pix.save(page_filename)
                                    logging.info(f"Uploading page {i} from PDF ({sender_phone}) to Gemini...")
                                    uploaded_file = genai.upload_file(path=page_filename, display_name=f"{sender_phone}_page_{i}.jpg")
                                    # Generate content for the specific page
                                    response = model.generate_content([prompt_prefix, uploaded_file])
                                    page_text = response.text # Use response.text for simplicity
                                    combined_doc_text += f"\n--- Page {i+1} ---\n{page_text}\n"
                                    remove_files(page_filename) # Clean up page file
                                    uploaded_file.delete() # Delete from Gemini storage
                                    uploaded_file = None # Reset tracker

                                # Send combined response after processing all pages
                                if combined_doc_text:
                                    user_convo.send_message(f"The user sent a PDF document. Here's a summary based on image analysis of its pages:\n{combined_doc_text}\n\n Respond to the user based on this summary.")
                                    reply = user_convo.last.text
                                    send_whatsapp_message(reply, sender_phone)
                                else:
                                    send_whatsapp_message("I processed the PDF, but couldn't extract meaningful content from the pages.", sender_phone)
                                return jsonify({"status": "ok"}), 200 # Exit after handling PDF

                            except Exception as e:
                                logging.error(f"Error processing PDF from {sender_phone}: {e}")
                                send_whatsapp_message("Sorry, I encountered an error while processing the PDF document.", sender_phone)
                                remove_files(f"/tmp/{sender_phone}_temp_doc_page_*") # Clean up any partial pages
                                if uploaded_file: uploaded_file.delete() # Clean up Gemini file if upload succeeded partially
                                return jsonify({"status": "error", "reason": "PDF processing failed"}), 200
                        else:
                            # Should not happen based on outer if, but good practice
                            send_whatsapp_message("This media type is not fully supported yet.", sender_phone)
                            return jsonify({"status": "ok"}), 200

                        # --- Common Handling for Image/Audio (after PDF is handled) ---
                        with open(filename, "wb") as temp_media:
                            temp_media.write(media_content)

                        logging.info(f"Uploading {message_type} ({sender_phone}) to Gemini...")
                        uploaded_file = genai.upload_file(path=filename, display_name=f"{sender_phone}_{message_type}")

                        # Generate content using the media
                        response = model.generate_content([prompt_prefix, uploaded_file])
                        media_description = response.text # Using response.text

                        # Clean up local file
                        remove_files(filename)

                        # Send description to the user's conversation context
                        user_convo.send_message(f"The user sent an {message_type}. The description is: '{media_description}'. Respond to the user based on this.")
                        reply = user_convo.last.text
                        send_whatsapp_message(reply, sender_phone)

                    else:
                        logging.warning(f"Unsupported message type '{message_type}' from {sender_phone}")
                        send_whatsapp_message("Sorry, I can only process text, images, audio, and PDF documents currently.", sender_phone)

                finally:
                    # Ensure Gemini file is deleted if it was uploaded
                    if uploaded_file:
                        try:
                            logging.info(f"Deleting uploaded file {uploaded_file.name} from Gemini.")
                            uploaded_file.delete()
                        except Exception as e:
                            logging.error(f"Failed to delete Gemini file {uploaded_file.name}: {e}")

            else:
                # Handle other types of notifications if needed (e.g., status updates)
                logging.info("Received non-message webhook or malformed data.")

        except Exception as e:
            logging.exception("Error processing webhook request:") # Logs traceback
            # Avoid crashing the webhook processor
            pass # Or return a specific error response if appropriate

        # Always return 200 OK to acknowledge receipt of the webhook
        return jsonify({"status": "ok"}), 200

    else:
        # Method Not Allowed
        return "Method Not Allowed", 405

if __name__ == "__main__":
    # Make sure essential env vars are present
    if not wa_token or not genai_api_key or not phone_id:
        logging.error("Missing one or more required environment variables: WA_TOKEN, GEN_API, PHONE_ID")
    else:
        port = int(os.environ.get("PORT", 8000)) # Use PORT env var if available (for deployment)
        app.run(host="0.0.0.0", port=port, debug=False) # Set debug=False for production

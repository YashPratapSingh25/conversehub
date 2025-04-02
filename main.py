from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import whisper
import os
import google.generativeai as genai
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the Whisper model
model_whisper = whisper.load_model("small", device="cpu")

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model_gemini = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    generation_config=generation_config,
    system_instruction="""

        Name: Damita

        Role: Conversational AI with Grammar, Style, and Paraphrasing

Persona: A friendly and encouraging language tutor, eager to help users improve their communication skills in a natural and conversational way.  Focuses on clarity, conciseness, and professionalism.

Important Point: Generate output in MD (Markdown)

Core Functions:

Conversational Continuation: Receives transcribed text from Whisper AI and generates relevant and engaging responses, maintaining context.

Grammar and Style Analysis: Analyzes the user's input (transcribed text) for:

Grammatical Errors (Excluding Punctuation): Identifies and flags grammatical mistakes (e.g., subject-verb agreement, tense errors, article usage). Punctuation errors are specifically ignored. Provides specific suggestions for correction.
Word Choice: Evaluates vocabulary, suggesting more precise, impactful, or varied alternatives.
Style and Tone: Assesses writing style and tone, offering feedback on how to adjust them.
Paraphrasing:  Generates a paraphrased version of the user's input, making it sound more professional and polished. This paraphrased version should:

Correct identified grammatical errors (excluding punctuation).
Incorporate suggested word choice improvements.
Maintain the original meaning while enhancing clarity and conciseness.
Contextual Awareness: Understands the ongoing conversation.

Adaptive Learning (Optional): Learns from user feedback.

If user sends "End Chat" then provide review of user's English skills and how much times he said correctly and what was his score for grammar, score for word selection and score for tone and also provide a level such as novice, advanced, pro according to his English and also provide where he can improve and how he can improve.

Interaction Flow:

Input: Transcribed text from Whisper AI is sent to the Gemini API.

Processing: The Gemini API performs:

Analysis of grammar (excluding punctuation), word choice, and style.
Generation of a paraphrased version of the input.
Generation of a conversational response.
Output: The Gemini API returns:

AI's conversational reply.
Detailed analysis of the transcribed text (grammar, word choice, style).
The paraphrased version of the user's input.
Presentation:

Display the AI's reply.
Display the paraphrased user input, perhaps highlighting the changes made (e.g., different color text). This allows the user to see how their message was refined.
Present the grammar and style analysis in a separate section.
Example Interaction:

User (via Whisper AI transcript): "me went to store yesterday.  I buyed some apples.  it was good."

Gemini API Output:

AI Reply: "That sounds like a pleasant trip!  What kind of apples did you buy?"

Paraphrased Input: "I went to the store yesterday and bought some apples. They were good."

Grammar Analysis:

"me went" - Subject-verb agreement error. Should be "I went."
"buyed" - Incorrect verb form. Should be "bought."
Word Choice:

"it was good" - Could be more descriptive. "They were delicious," "They were crisp and sweet," etc.
Style/Tone: Informal.

User Feedback: (User clicks on the paraphrased input)

Gemini API Explanation: "We've corrected the grammatical errors and made the sentence structure more standard.  We also kept the meaning the same while making the language more professional."
    """,
)

# Chat history storage
chat_history = []

class UserInput(BaseModel):
    text: str

@app.post("/analyze/")
async def analyze_text(user_input: UserInput):
    try:
        chat_session = model_gemini.start_chat(history=chat_history)
        response = chat_session.send_message(user_input.text)
        model_response = response.text
        
        chat_history.append({"role": "user", "parts": [user_input.text]})                                                                      
        chat_history.append({"role": "model", "parts": [model_response]})
         
        return {"response": model_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transcribe-and-analyze/")
async def transcribe_and_analyze(file: UploadFile = File(...)):
    try:
        audio_path = f"temp_{file.filename}"
        with open(audio_path, "wb") as f:
            f.write(await file.read())

        # Transcribe audio
        result = model_whisper.transcribe(audio_path, language="en")
        os.remove(audio_path)
        transcribed_text = result.get("text", "No text detected")
        
        # Analyze transcribed text
        analysis_result = await analyze_text(UserInput(text=transcribed_text))
        
        return {"transcription": transcribed_text, "analysis": analysis_result["response"]}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
@app.post("/end-chat/")
async def end_chat():
    try:
        if not chat_history:
            return {"message": "No conversation history to analyze."}

        # Send chat history to Gemini for summary
        chat_session = model_gemini.start_chat(history=chat_history)
        response = chat_session.send_message("End Chat")

        final_feedback = response.text

        # Clear chat history after analysis
        chat_history.clear()

        return {"feedback": final_feedback}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
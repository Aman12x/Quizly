from flask import Flask, request, jsonify
import os
from src.utils import *

# Initialize the Flask app
app = Flask(__name__)

# API Endpoints


# Serve the HTML form
@app.route("/")
def index():
    return """
        <!doctype html>
        <html>
        <head>
            <title>Upload Video for Summarization and Quiz</title>
        </head>
        <body>
            <h1>Upload a Video File</h1>
            <form method="POST" action="/transcribe" enctype="multipart/form-data">
                <label for="video">Select a video file:</label><br><br>
                <input type="file" id="video" name="video"><br><br>
                <input type="submit" value="Summarize and Generate Quiz">
            </form>
        </body>
        </html>
    """


# API endpoint to handle video upload and transcription
@app.route("/transcribe", methods=["POST"])
def transcribe_video():
    """API endpoint to transcribe a video and generate quizzes."""
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files["video"]
    video_path = os.path.join("uploads", video_file.filename)
    audio_path = video_path.replace(".mp4", ".mp3")

    # Save video to disk
    video_file.save(video_path)

    try:
        # Step 1: Extract audio
        extract_audio_from_video(video_path, audio_path)

        # Step 2: Transcribe the audio
        transcript = transcribe_audio(audio_path)

        summary = create_summary(transcript)

        # Step 3: Generate quiz
        quiz = create_quiz(summary)

        # Cleanup uploaded video and audio files
        os.remove(video_path)
        os.remove(audio_path)

        return f"<h1>Summary and Quiz Generated</h1><br><h2>Summary:</h2><p>{summary}</p><br><h2>Quiz:</h2><pre>{quiz}</pre>"

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Create the uploads directory if it doesn't exist
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    app.run(debug=True)

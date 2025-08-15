import json
import os
import sys
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from dotenv import load_dotenv

from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

import whisper
from moviepy import VideoFileClip

load_dotenv()

# Initialize local Whisper model for transcription
print("Loading Whisper model...")
whisper_model = whisper.load_model("base")  # Options: tiny, base, small, medium, large

# Initialize Ollama model (make sure Ollama is running locally)
llm = Ollama(
    model="llama3.1",  # or llama2, codellama, mistral, etc.
    temperature=0,
)

print("✅ Open source models loaded successfully!")


def extract_audio_from_video(video_path, audio_path):
    """Extract audio from video and save it to audio_path."""
    try:
        video = VideoFileClip(video_path)
        audio = video.audio

        # Suppress MoviePy output
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            audio.write_audiofile(audio_path)

        audio.close()
        video.close()
        print(f"Audio extracted and saved to: {audio_path}")
    except Exception as e:
        print(f"Error extracting audio: {e}")
        raise


def transcribe_audio_chunk(chunk_path):
    """Transcribe a small chunk of audio using local Whisper."""
    try:
        print(f"Transcribing with Whisper: {chunk_path}")
        result = whisper_model.transcribe(chunk_path)
        return result["text"]
    except Exception as e:
        print(f"Error transcribing chunk {chunk_path}: {e}")
        return ""


def transcribe_audio(audio_path):
    """Transcribe the audio file using local Whisper, handling large files by splitting them."""
    max_duration = 1800  # 30 minutes - Whisper can handle long files but we split for memory efficiency
    chunk_duration = 600  # 10 minutes in seconds

    try:
        # Check file duration
        from moviepy import AudioFileClip

        temp_audio = AudioFileClip(audio_path)
        duration = temp_audio.duration
        temp_audio.close()

        print(f"Audio duration: {duration} seconds")

        # If duration exceeds the limit, split the audio into smaller chunks
        if duration > max_duration:
            print("Splitting audio into chunks for better processing...")

            # Load audio with moviepy
            audio_clip = AudioFileClip(audio_path)

            chunks = []
            chunk_num = 0

            for start_time in range(0, int(duration), chunk_duration):
                end_time = min(start_time + chunk_duration, duration)
                chunk_clip = audio_clip.subclipped(start_time, end_time)

                chunk_path = f"{audio_path[:-4]}_chunk{chunk_num}.wav"

                # Suppress MoviePy output for chunk creation
                with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                    chunk_clip.write_audiofile(chunk_path)

                chunk_clip.close()

                chunks.append(chunk_path)
                chunk_num += 1
                print(f"Saved chunk: {chunk_path}")

            audio_clip.close()

            # Transcribe all chunks
            transcript_text = ""
            for chunk_path in chunks:
                print(f"Transcribing chunk: {chunk_path}")
                chunk_transcript = transcribe_audio_chunk(chunk_path)
                if chunk_transcript:  # Only add if transcription was successful
                    transcript_text += chunk_transcript + " "

                # Clean up after transcription
                try:
                    os.remove(chunk_path)
                    print(f"Deleted chunk file: {chunk_path}")
                except OSError as e:
                    print(f"Warning: Could not delete {chunk_path}: {e}")

            print("All chunks transcribed and concatenated.")
            return transcript_text.strip()

        # If duration is within limits, proceed with normal transcription
        print("Transcribing entire file with Whisper...")
        return transcribe_audio_chunk(audio_path)

    except Exception as e:
        print(f"Error processing audio file: {e}")
        raise


# Define your desired schema for quiz questions
class QuizQuestion(BaseModel):
    question: str = Field(description="The quiz question.")
    options: dict = Field(
        description="Answer options as a dictionary with keys a, b, c, d."
    )
    correct_answer: str = Field(
        description="The correct answer key (e.g., 'a', 'b', 'c', 'd')."
    )
    explanation: str = Field(description="Explanation for why the answer is correct.")


# Set up the parser with the schema
parser = JsonOutputParser(pydantic_object=QuizQuestion)


def create_summary(transcript):
    """Create a summary of the transcript focusing on key concepts."""
    try:
        # Summary
        summarization_template = """
        You are a helpful data science assistant. Summarize the following transcript and extract key concepts.
        Strictly discuss the concepts that are being discussed in the transcript and not the transcript itself.
        Focus on the main technical concepts, key points, and important information.

        Transcript: {transcript}

        Please provide a clear, concise summary focusing on the key concepts and main points discussed.
        """

        summarization_prompt = PromptTemplate(
            template=summarization_template, input_variables=["transcript"]
        )

        # Chain the prompt and model
        chain = summarization_prompt | llm

        output = chain.invoke({"transcript": transcript})
        print("Summary created successfully.")
        return output

    except Exception as e:
        print(f"Error creating summary: {e}")
        raise


def create_quiz(summary):
    """Generate quiz questions based on the summary."""
    try:
        # Generate quiz with explicit JSON format instruction
        quiz_template = """
        You are a helpful data science and engineering expert tasked with creating challenging quiz questions.
        
        Based on the following summary, create exactly 10 quiz questions in valid JSON format.
        Each question should be challenging and test deep understanding of the concepts.

        Summary: {summary}

        Please respond with a JSON array containing 10 quiz question objects. Each object must have exactly these fields:
        - "question": The quiz question as a string
        - "options": A dictionary with keys "a", "b", "c", "d" and their corresponding answer choices
        - "correct_answer": The correct answer key ("a", "b", "c", or "d")
        - "explanation": A string explaining why the answer is correct

        Example format:
        [
          {{
            "question": "What is...",
            "options": {{
              "a": "Option A text",
              "b": "Option B text", 
              "c": "Option C text",
              "d": "Option D text"
            }},
            "correct_answer": "a",
            "explanation": "The answer is A because..."
          }}
        ]

        Respond only with valid JSON, no additional text.
        """

        prompt = PromptTemplate(template=quiz_template, input_variables=["summary"])

        # Get raw response from model
        chain = prompt | llm
        raw_response = chain.invoke({"summary": summary})

        print("Raw quiz response received from model.")

        # Try to parse the JSON response
        try:
            # Clean up the response - remove any markdown formatting
            clean_response = raw_response.strip()
            if clean_response.startswith("```json"):
                clean_response = clean_response[7:]
            if clean_response.endswith("```"):
                clean_response = clean_response[:-3]
            clean_response = clean_response.strip()

            questions = json.loads(clean_response)
            print("Quiz questions generated and parsed successfully.")
            return questions
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            print(f"Raw response: {raw_response[:500]}...")
            # Return a fallback structure
            return [
                {
                    "question": "Failed to generate quiz - please check the model response format",
                    "options": {"a": "Error", "b": "Error", "c": "Error", "d": "Error"},
                    "correct_answer": "a",
                    "explanation": f"JSON parsing failed: {e}",
                }
            ]

    except Exception as e:
        print(f"Error creating quiz: {e}")
        raise


def main():
    """Main function to demonstrate the workflow."""
    print("🚀 Open Source Audio Transcription and Quiz Generator")
    print("📋 Using: Whisper (transcription) + Ollama/Llama (text generation)")
    print("=" * 70)

    # Check if Ollama is available
    try:
        test_response = llm.invoke("Hello, are you working?")
        print("✅ Ollama model is responding correctly.")
    except Exception as e:
        print("❌ Error: Cannot connect to Ollama.")
        print("Please make sure Ollama is installed and running:")
        print("1. Install Ollama: https://ollama.ai/download")
        print("2. Pull a model: ollama pull llama3.1")
        print("3. Start Ollama: ollama serve")
        print(f"Error details: {e}")
        return

    # Get video file path from user
    video_path = input(
        "Enter the path to your video file (or press Enter to skip): "
    ).strip()

    if not video_path:
        print("\n📝 No video file provided. Here's how to use this script:")
        print("\n🔧 Prerequisites:")
        print("1. Install Ollama: https://ollama.ai/download")
        print("2. Pull a model: ollama pull llama3.1")
        print("3. Make sure Ollama is running: ollama serve")
        print("\n📹 Usage:")
        print("1. Place your video file in an accessible location")
        print("2. Run the script again and enter the full file path")
        print("3. Or call the functions directly in your code")
        print("\n💡 Supported models: llama3.1, llama2, codellama, mistral, etc.")
        return

    if not os.path.exists(video_path):
        print(f"❌ Error: File '{video_path}' not found.")
        return

    try:
        audio_path = "extracted_audio.wav"

        # Step 1: Extract audio from video
        print(f"\n🎵 Extracting audio from {video_path}...")
        extract_audio_from_video(video_path, audio_path)

        # Step 2: Transcribe audio
        print(f"\n🎙️  Transcribing audio with Whisper (local)...")
        transcript = transcribe_audio(audio_path)
        print(f"✅ Transcription complete. Length: {len(transcript)} characters")
        print(
            f"📝 Preview: {transcript[:200]}..."
            if len(transcript) > 200
            else f"📝 Full transcript: {transcript}"
        )

        # Step 3: Create summary
        print(f"\n📄 Creating summary with Ollama...")
        summary = create_summary(transcript)
        print(f"✅ Summary created:")
        print(f"📋 {summary}")

        # Step 4: Generate quiz
        print(f"\n❓ Generating quiz questions with Ollama...")
        quiz_questions = create_quiz(summary)
        print(f"✅ Quiz generated successfully!")
        print(f"🧠 Quiz Questions:")
        print(json.dumps(quiz_questions, indent=2))

        # Step 5: Save results
        results = {
            "transcript": transcript,
            "summary": summary,
            "quiz": quiz_questions,
            "metadata": {
                "whisper_model": "base",
                "llm_model": "llama3.1",
                "source_file": os.path.basename(video_path),
            },
        }

        output_file = (
            f"results_{os.path.splitext(os.path.basename(video_path))[0]}.json"
        )
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n💾 Results saved to: {output_file}")

        # Clean up
        if os.path.exists(audio_path):
            os.remove(audio_path)
            print(f"🧹 Cleaned up temporary audio file: {audio_path}")

    except Exception as e:
        print(f"❌ An error occurred: {e}")
        # Clean up on error
        if os.path.exists("extracted_audio.wav"):
            os.remove("extracted_audio.wav")

    print("\n🎉 Process completed with open source tools!")


if __name__ == "__main__":
    main()

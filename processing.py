import os
import json
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import tempfile

from moviepy import VideoFileClip, AudioFileClip


def extract_audio_from_video(video_path: str, audio_path: str) -> bool:
    """Extract audio from video file."""
    try:
        video = VideoFileClip(video_path)
        audio = video.audio
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            audio.write_audiofile(audio_path)
        audio.close()
        video.close()
        return True
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return False


def transcribe_audio_with_progress(audio_path: str, whisper_model) -> str:
    """Transcribe audio with Whisper model."""
    try:
        audio_clip = AudioFileClip(audio_path)
        duration = audio_clip.duration
        audio_clip.close()
        max_duration = 1800  # 30 minutes
        chunk_duration = 600  # 10 minutes
        if duration > max_duration:
            transcript_parts = []
            audio_clip = AudioFileClip(audio_path)
            for i, start_time in enumerate(range(0, int(duration), chunk_duration)):
                end_time = min(start_time + chunk_duration, duration)
                chunk_clip = audio_clip.subclipped(start_time, end_time)
                chunk_path = f"temp_chunk_{i}.wav"
                with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                    chunk_clip.write_audiofile(chunk_path)
                chunk_clip.close()
                result = whisper_model.transcribe(chunk_path)
                transcript_parts.append(result["text"])
                os.remove(chunk_path)
            audio_clip.close()
            return " ".join(transcript_parts)
        else:
            result = whisper_model.transcribe(audio_path)
            return result["text"]
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return None


def create_summary(transcript: str, llm) -> str:
    """Create summary from transcript using LLM."""
    from langchain_core.prompts import PromptTemplate

    template = """
    You are a helpful data science assistant. Summarize the following transcript and extract key concepts.
    Focus on the main technical concepts, key points, and important information discussed.
    Transcript: {transcript}
    Please provide a clear, concise summary focusing on the key concepts and main points.
    """
    prompt = PromptTemplate(template=template, input_variables=["transcript"])
    chain = prompt | llm
    summary = chain.invoke({"transcript": transcript})
    return summary


def create_quiz(summary: str, llm):
    """Create quiz questions from summary using LLM."""
    from langchain_core.prompts import PromptTemplate
    import json

    template = """
    Based on the following summary, create exactly 5 challenging quiz questions in valid JSON format.
    Summary: {summary}
    Respond with a JSON array containing 5 quiz question objects. Each object must have:
    - "question": The quiz question as a string
    - "options": A dictionary with keys "a", "b", "c", "d" and their answer choices
    - "correct_answer": The correct answer key ("a", "b", "c", or "d")
    - "explanation": A string explaining why the answer is correct
    Respond only with valid JSON, no additional text.
    """
    prompt = PromptTemplate(template=template, input_variables=["summary"])
    chain = prompt | llm
    raw_response = chain.invoke({"summary": summary})
    clean_response = raw_response.strip()
    if clean_response.startswith("```json"):
        clean_response = clean_response[7:]
    if clean_response.endswith("```"):
        clean_response = clean_response[:-3]
    clean_response = clean_response.strip()
    questions = json.loads(clean_response)
    return questions

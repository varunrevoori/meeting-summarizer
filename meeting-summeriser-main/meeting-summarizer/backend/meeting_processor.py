# backend/meeting_processor.py
# --- FULL CODE ---
import os
import subprocess
import math
import time
from pathlib import Path
import argparse
import logging
import re  # For regular expression parsing for placeholders
import base64 # For embedding images
import json # Although not explicitly used for saving timestamps now, good practice
import traceback # For detailed error logging

# --- Third-party Libraries ---
# Ensure these are installed via requirements.txt
try:
    import google.generativeai as genai
    from groq import Groq
    import cv2  # OpenCV
    import numpy as np
    from PIL import Image  # Pillow for image handling with Gemini
    from dotenv import load_dotenv
    from pydub import AudioSegment
    from pydub.utils import make_chunks
except ImportError as e:
    # Log error during import if a library is missing
    logging.critical(f"Missing required library: {e}. Please install dependencies from requirements.txt")
    raise # Stop execution if libraries are missing


# --- Configuration ---
# Load environment variables from .env file relative to this script's location
dotenv_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=dotenv_path) # Load early

# --- Setup Logging ---
# Configure logging setup robustly
log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s')

# Get root logger and remove existing handlers to prevent duplicates
root_logger_setup = logging.getLogger()
if root_logger_setup.hasHandlers():
    for handler in root_logger_setup.handlers[:]:
        try: handler.close()
        except Exception: pass
        root_logger_setup.removeHandler(handler)

# Add console handler (optional, good for CLI runs)
# console_handler = logging.StreamHandler()
# console_handler.setFormatter(log_formatter)
# root_logger_setup.addHandler(console_handler)

root_logger_setup.setLevel(log_level)
logger = logging.getLogger(__name__) # Use a logger specific to this module
logger.info(f"Backend logger initialized with level {log_level}")


# API Keys (Loaded from .env)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Constants
FRAME_EXTRACTION_INTERVAL_SECONDS = 10 # Interval for *initial* scan
OUTPUT_DIR_BASE = Path("./processing_output") # Base directory for outputs if run directly from CLI
TEMP_AUDIO_FILENAME = "temp_meeting_audio.mp3"
IMPORTANT_FRAMES_SUBDIR = "important_frames" # Subdirectory name for key visuals
MAX_RETRIES = 3
RETRY_DELAY = 5 # seconds


# ================================================
# --- START OF HELPER FUNCTION DEFINITIONS ---
# ================================================
# Ensure ALL these functions are present in your file

def configure_apis():
    """Configures and returns the Groq and Gemini API clients."""
    logger.debug("Configuring APIs...")
    if not GROQ_API_KEY:
        logger.error("GROQ_API_KEY not found in environment variables.")
        raise ValueError("Missing Groq API Key")
    if not GOOGLE_API_KEY:
        logger.error("GOOGLE_API_KEY not found in environment variables.")
        raise ValueError("Missing Google API Key")

    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        genai.configure(api_key=GOOGLE_API_KEY)
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        # Consider making model names constants or configurable
        gemini_vision_model = genai.GenerativeModel('gemini-1.5-flash-latest')
        gemini_text_model = genai.GenerativeModel('gemini-1.5-flash-latest')
        logger.info("Groq and Gemini APIs configured successfully.")
        return groq_client, gemini_vision_model, gemini_text_model, safety_settings
    except Exception as e:
        logger.error(f"Error configuring APIs: {e}", exc_info=True)
        raise # Re-raise the exception

def extract_audio(video_path: Path, audio_path: Path) -> bool:
    """Extracts audio from video using FFmpeg."""
    logger.info(f"Extracting audio from '{video_path}' to '{audio_path}'...")
    command = [
        'ffmpeg',
        '-i', str(video_path),
        '-vn',           # Disable video recording
        '-acodec', 'libmp3lame', # Audio codec
        '-ab', '192k',   # Audio bitrate
        '-ar', '44100',  # Audio sample rate
        '-y',            # Overwrite output file if it exists
        str(audio_path)
    ]
    try:
        # Use timeout? Capture stderr on failure?
        process = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        logger.info("Audio extraction successful.")
        logger.debug(f"FFmpeg stdout:\n{process.stdout}")
        logger.debug(f"FFmpeg stderr:\n{process.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error during audio extraction (return code {e.returncode}): {e}")
        logger.error(f"FFmpeg stderr:\n{e.stderr}")
        return False
    except FileNotFoundError:
        logger.error("FFmpeg command not found. Please ensure FFmpeg is installed and in the system's PATH.")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during audio extraction: {e}", exc_info=True)
        return False


def transcribe_audio_with_groq(audio_path: Path, groq_client: Groq, chunk_length_min: int = 10) -> str | None:
    """Transcribes audio using Groq API (Whisper) with chunking."""
    logger.info(f"Transcribing audio file: {audio_path} (chunk length: {chunk_length_min} min)")
    if not audio_path.exists():
        logger.error(f"Audio file not found for transcription: {audio_path}")
        return None

    full_transcript = []
    output_dir = audio_path.parent
    try:
        # Load audio within try block
        logger.debug(f"Loading audio file {audio_path} with pydub...")
        audio = AudioSegment.from_mp3(str(audio_path))
        logger.info(f"Audio duration: {len(audio) / 1000:.2f} seconds")

        chunk_length_ms = chunk_length_min * 60 * 1000
        if chunk_length_ms <= 0:
             logger.warning("Chunk length must be positive, defaulting to 10 minutes.")
             chunk_length_ms = 10 * 60 * 1000

        chunks = make_chunks(audio, chunk_length_ms)
        logger.info(f"Split audio into {len(chunks)} chunks.")
        if not chunks:
            logger.warning("Audio splitting resulted in zero chunks.")
            return "" # Return empty string if no chunks

        for i, chunk in enumerate(chunks):
            chunk_number = i + 1
            # Use process ID and chunk number for potentially more unique temp names
            pid = os.getpid()
            chunk_filename = output_dir / f"temp_audio_chunk_{pid}_{chunk_number}.mp3"
            logger.info(f"Processing chunk {chunk_number}/{len(chunks)}...") # Log chunk progress

            # Export chunk
            try:
                logger.debug(f"Exporting chunk {chunk_number} to {chunk_filename}")
                chunk.export(str(chunk_filename), format="mp3")
            except Exception as e:
                logger.error(f"Error exporting chunk {chunk_number}: {e}", exc_info=True)
                full_transcript.append(f"[Error exporting chunk {chunk_number}]")
                continue # Skip to next chunk

            # Transcribe chunk
            retries = 0
            chunk_transcript_text = None
            while retries < MAX_RETRIES:
                try:
                    logger.debug(f"Transcribing chunk {chunk_number} (attempt {retries+1})")
                    with open(chunk_filename, "rb") as chunk_file:
                        # Ensure Groq client is correctly passed and used
                        transcription = groq_client.audio.transcriptions.create(
                            file=(chunk_filename.name, chunk_file.read()),
                            model="whisper-large-v3", # Consider making model configurable
                        )
                    # Add extra check for empty text?
                    chunk_transcript_text = transcription.text if transcription else "[Transcription Error: Empty Response]"
                    logger.info(f"Successfully transcribed chunk {chunk_number}.")
                    break # Success, exit retry loop
                except Exception as e:
                    retries += 1
                    logger.warning(f"Error transcribing chunk {chunk_number} (attempt {retries}/{MAX_RETRIES}): {e}")
                    if "413" in str(e) or "exceeds maximum file size" in str(e):
                         logger.error(f"Chunk {chunk_number} is STILL too large even after splitting. API limit exceeded.")
                         chunk_transcript_text = f"[Error: Chunk {chunk_number} file size limit]"
                         break # Don't retry size errors
                    if retries >= MAX_RETRIES:
                        logger.error(f"Transcription failed for chunk {chunk_number} after {MAX_RETRIES} retries.")
                        chunk_transcript_text = f"[Error transcribing chunk {chunk_number}]"
                        break # Failed after retries
                    logger.info(f"Retrying chunk {chunk_number} in {RETRY_DELAY} seconds...")
                    time.sleep(RETRY_DELAY)

            if chunk_transcript_text:
                full_transcript.append(chunk_transcript_text)
            else:
                logger.warning(f"Chunk {chunk_number} produced no transcript text after processing.")
                full_transcript.append(f"[Error: No transcript for chunk {chunk_number}]")


            # Clean up the temporary chunk file
            try:
                chunk_filename.unlink(missing_ok=True)
                logger.debug(f"Removed temporary chunk file: {chunk_filename}")
            except OSError as e:
                logger.warning(f"Could not remove temporary chunk file {chunk_filename}: {e}")

        # --- End of chunk loop ---
        logger.info("Finished processing all audio chunks.")
        final_text = " ".join(full_transcript).strip()
        logger.debug(f"Final transcript length: {len(final_text)}")
        return final_text

    except FileNotFoundError:
         # pydub/ffmpeg dependency issue likely
         logger.error(f"Error loading audio file {audio_path}. Check file existence AND FFmpeg/FFprobe setup for Pydub.", exc_info=True)
         return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during audio processing/transcription: {e}", exc_info=True)
        return None


def extract_frames_for_initial_scan(video_path: Path, interval_seconds: int) -> list[tuple[int, np.ndarray]]:
    """Extracts frames at intervals for the initial AI scan."""
    logger.info(f"Extracting frames for initial scan from '{video_path}' every {interval_seconds} seconds...")
    frames = []
    vidcap = None # Initialize to None
    try:
        vidcap = cv2.VideoCapture(str(video_path))
        if not vidcap.isOpened():
            logger.error(f"Error: Could not open video file: {video_path}")
            return frames # Return empty list

        fps = vidcap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0:
             logger.warning(f"Could not determine video FPS for {video_path}. Using interval directly as frame count.")
             frame_interval = interval_seconds # Treat interval as frame skip count
        else:
             frame_interval = int(fps * interval_seconds)
             if frame_interval <= 0:
                 logger.warning(f"Calculated frame interval ({frame_interval}) is zero or negative. Extracting only the first frame.")
                 frame_interval = 0 # Special case to extract only first frame

        logger.debug(f"Video FPS: {fps}, Frame extraction interval: {frame_interval} frames (approx {interval_seconds}s)")

        frame_count = 0
        saved_frame_count = 0
        while True:
            success, image = vidcap.read()
            if not success:
                logger.info("End of video reached or cannot read frame.")
                break # End of video or read error

            # Determine if this frame should be extracted
            is_extraction_frame = False
            if frame_interval > 0:
                if frame_count % frame_interval == 0:
                    is_extraction_frame = True
            elif frame_count == 0: # Special case for frame_interval == 0 (extract first frame)
                 is_extraction_frame = True


            if is_extraction_frame:
                timestamp_ms = int(vidcap.get(cv2.CAP_PROP_POS_MSEC))
                # Make a copy of the image data if keeping it in memory
                frames.append((timestamp_ms, image.copy()))
                saved_frame_count += 1
                logger.debug(f"Extracted frame #{saved_frame_count} at {timestamp_ms}ms (frame count: {frame_count})")

            frame_count += 1

        logger.info(f"Finished initial frame scan. Extracted {saved_frame_count} frames.")

    except Exception as e:
        logger.error(f"An error occurred during initial frame extraction: {e}", exc_info=True)
        # Return potentially partially extracted frames
    finally:
        if vidcap and vidcap.isOpened():
            vidcap.release()
            logger.debug("Video capture released.")

    return frames


def analyze_initial_frames(extracted_frames: list[tuple[int, np.ndarray]], gemini_vision_model, safety_settings) -> list[tuple[int, str]]:
    """Analyzes the initially scanned frames using Gemini Vision."""
    visual_descriptions = []
    total_frames = len(extracted_frames)
    if total_frames == 0:
        logger.info("No frames provided for initial analysis.")
        return visual_descriptions

    logger.info(f"Analyzing {total_frames} initially scanned frames...")
    prompt = """Describe the key visual elements in this image, focusing *only* on significant content like diagrams, flowcharts, code snippets, text on screen, mathematical formulas, or detailed whiteboards. Ignore generic elements like people talking, standard room backgrounds, or simple presentation title slides unless they contain substantial information. If no significant visual content is present, respond *only* with the exact phrase 'No significant visual content.'"""

    for i, (timestamp_ms, frame_image) in enumerate(extracted_frames):
        frame_num_log = f"frame {i+1}/{total_frames} at {timestamp_ms}ms"
        logger.debug(f"Analyzing {frame_num_log}...")
        try:
            # Add check for empty frame image
            if frame_image is None or frame_image.size == 0:
                 logger.warning(f"Skipping empty frame image for {frame_num_log}")
                 continue

            # Check image dimensions (optional sanity check)
            height, width = frame_image.shape[:2]
            if height == 0 or width == 0:
                logger.warning(f"Skipping invalid dimensions ({width}x{height}) for {frame_num_log}")
                continue

            rgb_image = cv2.cvtColor(frame_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            logger.debug(f"Converted {frame_num_log} to PIL format.")

        except Exception as e:
            logger.error(f"Error converting {frame_num_log} to PIL: {e}", exc_info=True)
            visual_descriptions.append((timestamp_ms, "[Error converting frame]"))
            continue # Skip analysis for this frame

        retries = 0
        description = None
        analysis_success = False
        while retries < MAX_RETRIES:
            try:
                logger.debug(f"Sending {frame_num_log} to Gemini Vision (attempt {retries+1})")
                response = gemini_vision_model.generate_content(
                    [prompt, pil_image], safety_settings=safety_settings,
                    request_options={"timeout": 60} # Add timeout?
                )
                # response.resolve() # resolve() might not be needed or could block, test behavior
                logger.debug(f"Received response for {frame_num_log}")

                # Check for blocking first - needs careful attribute checking
                block_reason = None
                try: # Defensive access to prompt_feedback
                    if response.prompt_feedback and response.prompt_feedback.block_reason:
                        block_reason = response.prompt_feedback.block_reason
                except AttributeError:
                     pass # Expected if feedback attribute doesn't exist

                if block_reason:
                    logger.warning(f"Analysis BLOCKED for {frame_num_log}. Reason: {block_reason}")
                    description = f"[Analysis blocked: {block_reason}]"
                    analysis_success = True # Mark as processed (even though blocked)
                    break # Don't retry if blocked

                # Process successful response text
                desc_text = response.text.strip()
                if desc_text != "No significant visual content.":
                    logger.info(f"Significant visual content found in {frame_num_log}.")
                    description = desc_text
                else:
                    logger.info(f"No significant visual content detected in {frame_num_log}.")
                    description = None # Explicitly set to None for non-significant frames
                analysis_success = True
                break # Success (or non-significant)

            except Exception as e:
                retries += 1
                logger.warning(f"Error analyzing {frame_num_log} (attempt {retries}/{MAX_RETRIES}): {e}")
                # Log specific error types if possible (e.g., timeouts, API errors)
                if retries >= MAX_RETRIES:
                    logger.error(f"Frame analysis failed permanently for {frame_num_log}.", exc_info=True)
                    description = "[Error analyzing frame]"
                    # Still mark as success=True here because we handled the error state? Or False?
                    # Let's say analysis 'attempted' but failed.
                    analysis_success = False # Indicate final failure
                    break # Exit retry loop
                logger.info(f"Retrying {frame_num_log} in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
        # --- End Retry Loop ---

        if description: # Add description only if significant, blocked, or error
             visual_descriptions.append((timestamp_ms, description))

        # Add a smaller delay between API calls if needed, even 0.1 can help sometimes
        time.sleep(0.2)

    logger.info(f"Finished initial frame analysis. Got {len(visual_descriptions)} descriptions.")
    return visual_descriptions


def generate_initial_summary_with_placeholders(transcript: str, initial_visual_descriptions: list[tuple[int, str]], gemini_text_model, safety_settings) -> str | None:
    """Generates the first-pass summary, requesting image placeholders."""
    logger.info("Generating initial summary and requesting image placeholders...")

    # Filter out error/blocked descriptions before formatting
    valid_visuals = [
        (ts, desc) for ts, desc in initial_visual_descriptions
        if desc and not desc.startswith("[Error") and not desc.startswith("[Analysis blocked")
    ]
    formatted_visuals = "\n".join(
        [f"- Frame at ~{ts // 1000}s ({ts}ms): {desc}" for ts, desc in valid_visuals]
    ) if valid_visuals else "No significant visual elements were detected or successfully analyzed."

    # --- MODIFIED PROMPT ---
    prompt = f"""
    **Meeting Transcript:**
    ---
    {transcript}
    ---

    **Potentially Significant Visuals Detected in Initial Scan (with approximate timestamps):**
    ---
    {formatted_visuals}
    ---

    **Instructions:**
    Generate a structured and comprehensive meeting summary using Markdown. Based *only* on the transcript and the visual descriptions above, identify the most important points.

    **Output Format:** Include these sections:
    1.  **Overall Summary:** Brief paragraph.
    2.  **Key Discussion Points:** Bullet points.
    3.  **Important Visual Content Discussed:** Describe significant diagrams, charts, code, formulas, or text *that were actually discussed or presented* according to the transcript and visual scan. Explain their context. Use LaTeX for formulas (e.g., $\\alpha$).
        **CRITICAL INSTRUCTION:** When you reference a specific important visual (like a flowchart, code snippet, key diagram) in this section, determine the most representative timestamp *in seconds* from the 'Potentially Significant Visuals' list above (use the timestamp closest to when it was discussed, or the first listed if it spans multiple mentions). Immediately after describing the visual and mentioning its approximate time (e.g., 'around 125s', 'at 300s'), insert a unique placeholder tag like @@IMG_XXX@@ where XXX is the timestamp *in seconds* you identified (e.g., @@IMG_125@@, @@IMG_300@@). **Only insert placeholders for visuals you deem truly important and reference in this specific section.** Do not insert placeholders anywhere else.
        *Example:* "...The presenter showed a complex flowchart (around 300s) @@IMG_300@@ that detailed the user signup process..."
        *Example:* "...Key performance metrics were displayed on a chart (around 545s) @@IMG_545@@ showing a Q3 increase..."
    4.  **Action Items (if mentioned):** List tasks, owners, deadlines. State if none.
    5.  **Decisions Made (if any):** List key decisions. State if none.

    Focus on clarity, conciseness, and accuracy. Strictly follow the placeholder insertion rule for the 'Important Visual Content Discussed' section. Ensure the output is valid Markdown.
    """
    # --- END OF MODIFIED PROMPT ---

    retries = 0
    while retries < MAX_RETRIES:
        try:
            logger.debug(f"Generating initial summary (attempt {retries+1})...")
            response = gemini_text_model.generate_content(
                prompt,
                safety_settings=safety_settings,
                generation_config={"temperature": 0.4}, # Keep temperature reasonable
                request_options={"timeout": 120} # Longer timeout for generation
            )
            # response.resolve() # Test if needed

            # Check for blocking first
            block_reason = None
            try: # Defensive access
                if response.prompt_feedback and response.prompt_feedback.block_reason:
                    block_reason = response.prompt_feedback.block_reason
            except AttributeError:
                pass

            if block_reason:
                 logger.error(f"Initial summary generation BLOCKED. Reason: {block_reason}")
                 # Return the failure message directly
                 return f"# Summary Generation Failed\n\nGeneration was blocked due to safety settings: {block_reason}"

            # Check if text content exists
            if response.text:
                logger.info("Initial summary generation successful.")
                return response.text
            else:
                 # Handle case where response is successful but has no text (unlikely but possible)
                 logger.error("Initial summary generation succeeded but returned empty text.")
                 raise ValueError("Summary generation returned empty content")

        except Exception as e:
            retries += 1
            logger.warning(f"Error during initial summary generation (attempt {retries}/{MAX_RETRIES}): {e}")
            if retries >= MAX_RETRIES:
                logger.error("Initial summary generation failed after multiple retries.", exc_info=True)
                return None # Return None on final failure
            logger.info(f"Retrying summary generation in {RETRY_DELAY} seconds...")
            time.sleep(RETRY_DELAY)
    # Should only be reached if loop finishes without success or error block being hit (unlikely)
    logger.error("Summary generation loop completed unexpectedly.")
    return None


def parse_placeholders(summary_text: str) -> dict[str, int]:
    """Parses @@IMG_XXX@@ placeholders and returns a dict {placeholder: timestamp_ms}."""
    logger.debug("Parsing summary text for placeholders...")
    placeholders = {}
    if not summary_text: # Handle empty summary case
        logger.warning("Cannot parse placeholders from empty summary text.")
        return placeholders

    # Pattern finds @@IMG_ followed by digits (captured), followed by @@
    pattern = r"@@IMG_(\d+)@@"
    try:
        matches = re.finditer(pattern, summary_text)
        count = 0
        for match in matches:
            count += 1
            placeholder_tag = match.group(0) # Full tag e.g., @@IMG_125@@
            seconds_str = match.group(1)     # Captured digits e.g., "125"
            try:
                timestamp_ms = int(seconds_str) * 1000
                if placeholder_tag not in placeholders: # Store first occurrence
                    placeholders[placeholder_tag] = timestamp_ms
                    logger.info(f"Found placeholder: {placeholder_tag} requesting frame at {timestamp_ms}ms")
                else:
                     logger.warning(f"Duplicate placeholder tag found and ignored: {placeholder_tag}")
            except ValueError:
                logger.warning(f"Could not parse timestamp '{seconds_str}' from placeholder: {placeholder_tag}")
        if count == 0:
            logger.info("No valid @@IMG_XXX@@ placeholders found in the summary.")
        else:
             logger.info(f"Parsed {len(placeholders)} unique placeholders.")
    except Exception as e:
         logger.error(f"Error during regex parsing for placeholders: {e}", exc_info=True)
         # Return empty dict on regex error
         return {}
    return placeholders


def extract_specific_frames(video_path: Path, timestamps_ms: list[int], output_dir: Path) -> dict[int, Path]:
    """
    Extracts specific frames based on timestamps (ms) and saves them.
    Returns a dictionary mapping {timestamp_ms: path_to_saved_frame} for successfully extracted frames.
    """
    saved_frames = {}
    if not timestamps_ms:
        logger.info("No specific timestamps provided for frame extraction.")
        return saved_frames

    # Ensure the output directory for important frames exists
    frames_output_dir = output_dir / IMPORTANT_FRAMES_SUBDIR
    try:
        frames_output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
         logger.error(f"Failed to create important frames directory {frames_output_dir}: {e}", exc_info=True)
         return saved_frames # Cannot save frames

    logger.info(f"Attempting to extract {len(timestamps_ms)} specific frames into '{frames_output_dir}'...")

    vidcap = None
    extracted_count = 0
    # Sort and unique timestamps
    unique_timestamps = sorted(list(set(timestamps_ms)))
    logger.debug(f"Unique timestamps requested: {unique_timestamps}")

    try:
        vidcap = cv2.VideoCapture(str(video_path))
        if not vidcap.isOpened():
            logger.error(f"Error opening video file for specific frame extraction: {video_path}")
            return saved_frames

        for ts_ms in unique_timestamps:
            logger.debug(f"Seeking to {ts_ms}ms...")
            # Seeking can be inaccurate, might need multiple reads or seeking slightly before
            seek_success = vidcap.set(cv2.CAP_PROP_POS_MSEC, ts_ms)
            if not seek_success:
                # Some video formats might report false for seek success even if it works.
                logger.warning(f"Seek operation to timestamp {ts_ms}ms reported failure (may still work).")
                # Continue attempt regardless?

            # Read the frame *after* seeking. It might not be the *exact* timestamp.
            success, frame = vidcap.read()

            if success:
                actual_timestamp_ms = int(vidcap.get(cv2.CAP_PROP_POS_MSEC)) # Get actual position
                logger.debug(f"Read frame successfully after seeking to {ts_ms}ms. Actual position: {actual_timestamp_ms}ms")
                frame_filename = frames_output_dir / f"frame_at_{ts_ms}ms.jpg" # Use requested ts for filename consistency
                try:
                    # Add quality parameter for JPG
                    save_success = cv2.imwrite(str(frame_filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    if save_success:
                        logger.info(f"Successfully extracted frame for requested {ts_ms}ms (actual {actual_timestamp_ms}ms) to {frame_filename}")
                        # Store mapping using the *requested* timestamp as key for lookup later
                        saved_frames[ts_ms] = frame_filename
                        extracted_count += 1
                    else:
                         logger.error(f"cv2.imwrite failed to save frame for timestamp {ts_ms}ms to {frame_filename} (returned false).")

                except Exception as e:
                    logger.error(f"Exception saving frame for timestamp {ts_ms}ms: {e}", exc_info=True)
            else:
                # Reading after seek failed. Maybe timestamp is beyond video length?
                logger.warning(f"Failed to READ frame after seeking to timestamp {ts_ms}ms. Might be end of video or issue.")

    except Exception as e:
        logger.error(f"An error occurred during specific frame extraction loop: {e}", exc_info=True)
    finally:
        if vidcap and vidcap.isOpened():
            vidcap.release()
            logger.debug("Video capture released after specific frame extraction.")

    logger.info(f"Finished specific frame extraction. Successfully extracted {extracted_count}/{len(unique_timestamps)} requested frames.")
    return saved_frames


def encode_extracted_frames(saved_frames_map: dict[int, Path]) -> dict[int, str]:
    """Encodes the saved frames into Base64 strings."""
    base64_map = {}
    if not saved_frames_map:
        logger.info("No saved frames provided for encoding.")
        return base64_map

    logger.info(f"Encoding {len(saved_frames_map)} extracted frames to Base64...")
    for ts_ms, frame_path in saved_frames_map.items():
        logger.debug(f"Encoding frame for {ts_ms}ms from {frame_path}")
        try:
            if frame_path.is_file():
                with open(frame_path, "rb") as img_file:
                    img_data = img_file.read()
                if not img_data:
                     logger.warning(f"Frame file {frame_path} for {ts_ms}ms is empty.")
                     continue # Skip empty file
                b64_img = base64.b64encode(img_data).decode('utf-8')
                base64_map[ts_ms] = b64_img
                logger.debug(f"Successfully encoded frame for {ts_ms}ms.")
            else:
                logger.warning(f"Frame file {frame_path} for timestamp {ts_ms}ms not found during encoding phase.")
        except Exception as e:
            logger.error(f"Could not read or encode image {frame_path} for timestamp {ts_ms}ms: {e}", exc_info=True)
            # Do not add to map if encoding failed

    logger.info(f"Finished encoding frames. {len(base64_map)} frames successfully encoded.")
    return base64_map


def replace_placeholders_with_images(summary_text: str, placeholder_map: dict[str, int], base64_image_map: dict[int, str]) -> str:
    """Replaces placeholders in the summary text with Base64 encoded images or error notes."""
    if not placeholder_map:
        logger.info("No placeholders found in summary, skipping replacement.")
        return summary_text # Return original text if no placeholders

    logger.info("Replacing placeholders with embedded images in the final summary...")
    final_summary = summary_text
    replaced_count = 0
    missed_count = 0

    # Iterate through the unique placeholders found earlier
    for placeholder_tag, ts_ms in placeholder_map.items():
        logger.debug(f"Processing placeholder: {placeholder_tag} for timestamp {ts_ms}ms")
        replacement_text = ""
        if ts_ms in base64_image_map:
            # Image data exists for this placeholder's timestamp
            b64_img_data = base64_image_map[ts_ms]
            alt_text = f"Visual content at {ts_ms // 1000}s ({ts_ms}ms)"
            # Standard Markdown for embedding Base64 images
            # Ensure newlines for proper rendering
            replacement_text = f"\n\n![{alt_text}](data:image/jpeg;base64,{b64_img_data})\n\n"
            logger.debug(f"Replacing {placeholder_tag} with embedded image for {ts_ms}ms.")
            replaced_count += 1
        else:
            # Image data is missing (extraction failed or encoding failed)
            error_note = f"*<Visual content requested for {ts_ms // 1000}s ({ts_ms}ms) could not be embedded>*"
            replacement_text = f"\n\n{error_note}\n\n"
            logger.warning(f"Replacing {placeholder_tag} with an error note (no Base64 data found for {ts_ms}ms).")
            missed_count += 1

        # Replace all occurrences of this specific tag. Use count=1 if tags should be unique.
        final_summary = final_summary.replace(placeholder_tag, replacement_text)

    logger.info(f"Placeholder replacement complete. Embedded: {replaced_count}, Missed: {missed_count}.")
    return final_summary


# ==============================================
# --- END OF HELPER FUNCTION DEFINITIONS ---
# ==============================================


# ============================================================
# --- MAIN GENERATOR FUNCTION (Called by Streamlit App) ---
# ============================================================
def process_video_step_by_step(
    video_path: Path,
    output_dir: Path,
    frame_interval: int = 10,
    audio_chunk_minutes: int = 10
    ):
    """
    Processes video yielding status updates and finally the summary path or raises Exception.

    Yields:
        tuple[str, float]: (Status message, Progress percentage (0.0 to 1.0))
        Path: The final summary file path on success.
    """
    total_steps = 11 # Approximate number of major stages for progress calculation
    current_step = 0

    # --- Internal Progress Helper ---
    def _update_progress(step, message):
        nonlocal current_step
        current_step = step
        # Calculate progress, ensuring it doesn't exceed 1.0
        progress = min(1.0, round(max(0.0, current_step) / total_steps, 2))
        logger.info(f"PROGRESS ({progress*100:.0f}%): {message}")
        return (message, progress)

    logger.debug(f"--- process_video_step_by_step called ---")
    logger.debug(f"Video path: {video_path}, Output dir: {output_dir}, Frame Interval: {frame_interval}, Chunk Min: {audio_chunk_minutes}")

    if not video_path.is_file():
        msg = f"Video file not found or is not a file: {video_path}"
        logger.error(msg)
        raise ValueError(msg) # Raise exception for Streamlit to catch

    # --- Step 0: Ensure Output Dir ---
    yield _update_progress(0, "Initializing: Preparing output directory...")
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Output directory ensured: {output_dir}")
    except Exception as e:
        msg = f"Failed to create output directory {output_dir}: {e}"
        logger.error(msg, exc_info=True)
        raise IOError(msg) # Raise exception

    # Define paths
    temp_audio_path = output_dir / TEMP_AUDIO_FILENAME
    summary_file_path = output_dir / f"{Path(video_path.stem).name}_summary_with_images.md" # Use original stem name
    logger.info(f"Starting multimodal summarization. Output target: {summary_file_path}")

    # --- Step 1: Configure APIs ---
    yield _update_progress(1, "Initializing: Configuring APIs...")
    try:
        groq_client, gemini_vision_model, gemini_text_model, safety_settings = configure_apis()
    except Exception as e:
        msg = f"API configuration failed: {e}"
        logger.error(msg, exc_info=True)
        raise ConnectionError(msg) # Raise exception

    # --- Step 2: Extract Audio ---
    yield _update_progress(2, f"Processing: Extracting audio...")
    try:
        audio_extracted = extract_audio(video_path, temp_audio_path)
        if not audio_extracted:
            # extract_audio logs the specific error (FFmpeg missing or execution error)
            msg = "Audio extraction failed (check FFmpeg setup and video file integrity)."
            logger.error(msg)
            raise RuntimeError(msg) # Raise exception
    except Exception as e: # Catch unexpected errors during the call
        msg = f"Unexpected error during audio extraction call: {e}"
        logger.error(msg, exc_info=True)
        raise RuntimeError(msg)


    # --- Step 3: Transcribe Audio ---
    yield _update_progress(3, "Processing: Starting audio transcription...")
    transcript = None
    try:
        # Pass helper functions here if they needed finer-grained progress yields
        transcript = transcribe_audio_with_groq(temp_audio_path, groq_client, chunk_length_min=audio_chunk_minutes)
        if transcript is None: # Check for None specifically
             # transcription function should log the specific error
             raise RuntimeError("Audio transcription returned None (check logs for details).")
        yield _update_progress(4, "Processing: Audio transcription complete.") # Mark end of this phase
        logger.info(f"Transcription complete. Length: {len(transcript)} chars.")
    except Exception as e:
        msg = f"Audio transcription step failed: {e}"
        logger.error(msg, exc_info=True)
        raise RuntimeError(msg) # Raise exception
    finally:
        # Ensure temp audio is cleaned up even if transcription fails
        try:
            if temp_audio_path.exists():
                temp_audio_path.unlink()
                logger.debug(f"Cleaned up temp audio file: {temp_audio_path}")
        except OSError as e:
            logger.warning(f"Could not remove temp audio file {temp_audio_path} during cleanup: {e}")

    # --- Step 4: Extract Initial Frames ---
    yield _update_progress(5, f"Processing: Extracting initial frames...")
    initial_frames = extract_frames_for_initial_scan(video_path, frame_interval)

    # --- Step 5: Analyze Initial Frames ---
    yield _update_progress(6, f"Processing: Analyzing {len(initial_frames)} initial frames...")
    initial_visual_descriptions = []
    if initial_frames:
        try:
            initial_visual_descriptions = analyze_initial_frames(initial_frames, gemini_vision_model, safety_settings)
            logger.info(f"Initial frame analysis complete. Found {len(initial_visual_descriptions)} descriptions.")
        except Exception as e:
             logger.error(f"Error during initial frame analysis step: {e}", exc_info=True)
             # Option 1: Continue without visual info
             # yield _update_progress(7, "Warning: Initial frame analysis failed, continuing...")
             # Option 2: Fail hard
             raise RuntimeError(f"Initial frame analysis failed: {e}")
    else:
        logger.info("No initial frames to analyze.")
    # Always yield step 7 completion, even if no frames were analyzed
    yield _update_progress(7, "Processing: Initial frame analysis complete.")

    # --- Step 6: Generate Initial Summary ---
    yield _update_progress(8, "Processing: Generating summary draft...")
    initial_summary_text = None
    try:
        initial_summary_text = generate_initial_summary_with_placeholders(
            transcript, initial_visual_descriptions, gemini_text_model, safety_settings
        )
        # Check if generation failed (e.g., due to blocking or API error)
        if initial_summary_text is None or "Summary Generation Failed" in initial_summary_text:
            err_msg = initial_summary_text or 'No content returned'
            logger.error(f"Summary draft generation failed: {err_msg}")
            raise RuntimeError(f"Summary draft generation failed: {err_msg}")
        logger.info("Summary draft generated successfully.")
    except Exception as e:
        msg = f"Failed to generate summary draft: {e}"
        logger.error(msg, exc_info=True)
        raise RuntimeError(msg) # Raise exception

    # --- Step 7: Parse Placeholders ---
    yield _update_progress(9, "Processing: Identifying key visuals requested...")
    placeholder_map = parse_placeholders(initial_summary_text)

    # --- Step 8 & 9: Extract & Encode Specific Frames ---
    base64_image_map = {}
    required_timestamps_ms = list(placeholder_map.values())
    if required_timestamps_ms:
        # If placeholders exist, this step takes significant time
        yield _update_progress(10, f"Processing: Extracting & encoding {len(required_timestamps_ms)} key frames...")
        try:
            extracted_frames_map = extract_specific_frames(video_path, required_timestamps_ms, output_dir)
            if extracted_frames_map:
                base64_image_map = encode_extracted_frames(extracted_frames_map)
                logger.info(f"Extracted and encoded {len(base64_image_map)} key frames.")
            else:
                # This isn't necessarily an error, could just be seek issues
                logger.warning("Specific frame extraction yielded no frames despite requests.")
        except Exception as e:
             logger.error(f"Error during specific frame extraction/encoding: {e}", exc_info=True)
             # Decide whether to proceed or fail
             yield _update_progress(10, "Warning: Key frame extraction failed...") # Yield intermediate warning
             # Option: Continue without images for these placeholders
             # Option: raise RuntimeError(f"Key frame processing failed: {e}") # Fail hard
    else:
        # If no placeholders, this step is quick
        yield _update_progress(10, "Processing: No specific frames requested.") # Mark step complete
        logger.debug("No specific frames needed.")

    # --- Step 10: Replace Placeholders ---
    # This is usually CPU-bound and fast, merge progress reporting
    logger.debug("Processing: Embedding images into summary...")
    try:
        final_summary_content = replace_placeholders_with_images(
            initial_summary_text, placeholder_map, base64_image_map
        )
    except Exception as e:
         msg = f"Failed to replace placeholders in summary: {e}"
         logger.error(msg, exc_info=True)
         raise RuntimeError(msg) # Raise exception

    # --- Step 11: Save Final Summary ---
    yield _update_progress(11, "Finalizing: Saving summary...")
    try:
        with open(summary_file_path, "w", encoding="utf-8") as f:
            f.write(final_summary_content)
        logger.info(f"Final summary saved successfully to {summary_file_path}")
        logger.debug("--- process_video_step_by_step finished successfully ---")

        # >>> FINAL YIELD: Return the path on success <<<
        yield summary_file_path

    except IOError as e:
        msg = f"Error writing final summary file {summary_file_path}: {e}"
        logger.error(msg, exc_info=True)
        raise IOError(msg) # Raise exception


# ============================================================
# --- Command Line Execution Block (for direct testing) ---
# ============================================================
if __name__ == "__main__":
    # Setup basic console logging specific to CLI run
    cli_log_level_str = os.environ.get('LOG_LEVEL', 'INFO').upper()
    cli_log_level = getattr(logging, cli_log_level_str, logging.INFO) # Convert string to level

    # Configure root logger for CLI specifically
    cli_root_logger = logging.getLogger()
    if cli_root_logger.hasHandlers():
        for handler in cli_root_logger.handlers[:]:
            try: handler.close()
            except Exception: pass
            cli_root_logger.removeHandler(handler)
    cli_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [CLI:%(lineno)d] - %(message)s')
    cli_console_handler = logging.StreamHandler()
    cli_console_handler.setFormatter(cli_formatter)
    cli_root_logger.addHandler(cli_console_handler)
    cli_root_logger.setLevel(cli_log_level)

    logger.info("Running backend script directly from command line...")

    parser = argparse.ArgumentParser(description="Summarize meeting video with embedded important visuals (CLI).")
    parser.add_argument("video_file", help="Path to the video meeting recording file.")
    parser.add_argument("-i", "--interval", type=int, default=FRAME_EXTRACTION_INTERVAL_SECONDS,
                        help=f"Interval (seconds) for *initial* frame scan (default: {FRAME_EXTRACTION_INTERVAL_SECONDS})")
    parser.add_argument("-o", "--output_dir", default=str(OUTPUT_DIR_BASE),
                        help=f"Base directory for output files (default: {OUTPUT_DIR_BASE})")
    parser.add_argument("--chunk_minutes", type=int, default=10,
                        help="Chunk size in minutes for audio transcription (default: 10)")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging for CLI run.")
    args = parser.parse_args()

    # Override log level if debug flag is set
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("DEBUG logging enabled via --debug flag.")

    # Prepare paths
    video_path_arg = Path(args.video_file)
    if not video_path_arg.is_file():
         logger.error(f"Video file not found: {video_path_arg}")
         exit(1)

    cli_output_base = Path(args.output_dir)
    # Create a unique subdir for this CLI run within the base output dir
    run_output_dir = cli_output_base / f"{video_path_arg.stem}_cli_{time.strftime('%Y%m%d_%H%M%S')}"
    try:
        run_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"CLI Run Output Directory: {run_output_dir}")
    except Exception as e:
        logger.error(f"Failed to create CLI output directory {run_output_dir}: {e}", exc_info=True)
        exit(1) # Exit if cannot create output dir

    # Ensure .env is loaded from script's parent directory if run directly
    if not load_dotenv(dotenv_path=Path(__file__).parent / '.env'):
         logger.warning("Could not find .env file in backend directory for CLI run.")
    else:
        logger.info(".env file loaded for CLI run.")


    # --- Execute using the generator ---
    final_path = None
    try:
        logger.info("Starting processing via generator for CLI...")
        # Iterate through the generator results
        for cli_result in process_video_step_by_step(
            video_path=video_path_arg,
            output_dir=run_output_dir,
            frame_interval=args.interval,
            audio_chunk_minutes=args.chunk_minutes
        ):
            if isinstance(cli_result, Path):
                final_path = cli_result # Store the final path
                logger.info("Processing finished successfully.")
                break # Stop iteration
            elif isinstance(cli_result, tuple) and len(cli_result) == 2:
                cli_msg, cli_prog = cli_result
                # Print progress to console for CLI runs
                print(f"PROGRESS: {cli_msg} ({cli_prog*100:.0f}%)", flush=True)
            else:
                logger.warning(f"Unexpected yield type in CLI loop: {type(cli_result)}")

        # Check result after loop
        if final_path:
            print(f"\n✅ Processing complete.")
            print(f"   Summary saved to: {final_path.resolve()}") # Show absolute path
            important_frames_dir = final_path.parent / IMPORTANT_FRAMES_SUBDIR
            if important_frames_dir.exists():
                print(f"   Extracted frames saved in: {important_frames_dir.resolve()}") # Show absolute path
        else:
             # Generator finished without error but didn't yield path? Should not happen.
             print("\n⚠️ Processing finished, but no summary file path was returned.")
             exit_code = 1 # Indicate potential issue

    except Exception as e:
        print(f"\n❌ Processing failed with an error: {e}")
        # Logger should have already logged the traceback if configured correctly
        print("   Check log messages above or log file for more details.")
        exit_code = 1 # Indicate failure

    exit(exit_code if 'exit_code' in locals() else 0) # Exit with appropriate code
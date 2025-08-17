from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from tempfile import NamedTemporaryFile
import subprocess
import os
import uuid
import json
import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

WHISPER_MODEL_FMT = "./whisper.cpp/models/ggml-{model}.{language}.bin"
WHISPER_BINARY = os.getenv("WHISPER_BINARY", "./whisper.cpp/build/bin/whisper-cli")
DEFAULT_MODEL = "small"  # Default model to use when whisper-1 is specified


async def _prepare_wav_input(file: UploadFile):
    """
    Prepares a WAV file for transcription.
    If the input file is not WAV, it converts it to WAV using ffmpeg.
    Returns the path to the temporary WAV file and the path of the original temp file if conversion occurred.
    """
    original_temp_name = None
    input_wav_path = None

    # Check if it's already a WAV file
    if file.content_type in ["audio/wav", "audio/x-wav"]:
        with NamedTemporaryFile(delete=False, suffix=".wav") as temp_input_wav:
            temp_input_wav.write(await file.read())
            temp_input_wav.flush()
            input_wav_path = temp_input_wav.name
    else:
        # Handle other audio formats (WebM, MP3, MP4, etc.)
        # Determine file extension from filename or content type
        if file.filename:
            original_suffix = os.path.splitext(file.filename)[1]
        elif file.content_type:
            # Map common content types to extensions
            content_type_to_ext = {
                "audio/webm": ".webm",
                "audio/mp4": ".m4a",
                "audio/mpeg": ".mp3",
                "audio/ogg": ".ogg",
                "audio/flac": ".flac",
                "audio/aac": ".aac"
            }
            original_suffix = content_type_to_ext.get(file.content_type, ".tmp")
        else:
            original_suffix = ".tmp"
        
        with NamedTemporaryFile(
            delete=False, suffix=original_suffix
        ) as temp_original_file:
            temp_original_file.write(await file.read())
            temp_original_file.flush()
            original_temp_name = temp_original_file.name

        with NamedTemporaryFile(delete=False, suffix=".wav") as temp_converted_wav:
            input_wav_path = temp_converted_wav.name

        # FFmpeg command to convert any audio format to WAV
        # -f format detection is automatic, so we don't need to specify input format
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file
            "-i", original_temp_name,  # Input file
            "-ar", "16000",  # Sample rate: 16kHz
            "-ac", "1",      # Channels: mono
            "-c:a", "pcm_s16le",  # Audio codec: 16-bit PCM
            input_wav_path,
        ]
        
        logging.info(f"Converting {file.content_type or 'unknown'} file to WAV: {' '.join(ffmpeg_cmd)}")
        
        try:
            process_handle = subprocess.run(
                ffmpeg_cmd, check=True, capture_output=True, text=True
            )
            logging.info(f"FFmpeg conversion successful: {process_handle.stdout}")
        except subprocess.CalledProcessError as e:
            logging.error(f"FFmpeg conversion failed: {e.stderr}")
            raise

    return input_wav_path, original_temp_name


@app.post("/v1/audio/transcriptions")
async def transcribe_audio(
    file: UploadFile = File(...),
    model: str = Form("base"),
    language: str = Form("en"),
):
    input_wav_path = None
    original_temp_file_path = (
        None  # To store path of original uploaded file if conversion happens
    )
    output_prefix = f"data/out-{uuid.uuid4()}"

    try:
        input_wav_path, original_temp_file_path = await _prepare_wav_input(file)
        # Use DEFAULT_MODEL if whisper-1 is specified
        actual_model = DEFAULT_MODEL if model == "whisper-1" else model
        model_path = WHISPER_MODEL_FMT.format(model=actual_model, language=language)
        if not os.path.exists(model_path):
            return JSONResponse(
                status_code=400,
                content={"error": f"Model '{model}' for language '{language}' does not exist. Please check the model name and language."},
            )

        cmd = [
            WHISPER_BINARY,
            "-m",
            model_path,
            "-f",
            input_wav_path,
            "-ojf",
            "-of",
            output_prefix,
            "-l",
            language,
        ]

        logging.info(" ".join(cmd))
        
        # Run whisper with text=False first to capture raw bytes, then decode safely
        process_handle = subprocess.run(cmd, check=True, capture_output=True, text=False)
        
        # Safely decode stdout and stderr
        try:
            stdout_text = process_handle.stdout.decode('utf-8', errors='replace') if process_handle.stdout else ""
            stderr_text = process_handle.stderr.decode('utf-8', errors='replace') if process_handle.stderr else ""
        except Exception as decode_e:
            logging.error(f"Error decoding process output: {decode_e}")
            stdout_text = str(process_handle.stdout) if process_handle.stdout else ""
            stderr_text = str(process_handle.stderr) if process_handle.stderr else ""
        
        logging.info(f"Whisper stdout: {stdout_text}")
        logging.info(f"Whisper stderr: {stderr_text}")
        
        # Read the JSON file with explicit encoding
        json_file_path = f"{output_prefix}.json"
        logging.info(f"Reading JSON file: {json_file_path}")
        
        try:
            with open(json_file_path, "r", encoding="utf-8") as out_file:
                file_content = out_file.read()
                logging.info(f"JSON file content length: {len(file_content)}")
                transcription_data = json.loads(file_content)
                transcription = transcription_data["transcription"][0]["text"]
        except UnicodeDecodeError as e:
            logging.error(f"Unicode decode error reading JSON file: {e}")
            # Try to read as bytes and decode with error handling
            with open(json_file_path, "rb") as out_file:
                file_content = out_file.read()
                logging.info(f"JSON file bytes length: {len(file_content)}")
                # Try to decode with error handling
                try:
                    decoded_content = file_content.decode('utf-8', errors='replace')
                    transcription_data = json.loads(decoded_content)
                    transcription = transcription_data["transcription"][0]["text"]
                    logging.warning("Successfully decoded JSON with replacement characters")
                except Exception as decode_e:
                    logging.error(f"Failed to decode JSON even with replacement: {decode_e}")
                    raise
    except subprocess.CalledProcessError as e:
        error_message = "Whisper CLI failed"
        if e.cmd:
            if e.cmd[0] == "ffmpeg":
                error_message = f"ffmpeg conversion failed: {e.stderr if e.stderr else 'Unknown error'}"
                logging.error(f"FFmpeg conversion failed for file {file.filename} (content-type: {file.content_type}): {e.stderr if e.stderr else 'Unknown error'}")
            elif e.cmd[0] == WHISPER_BINARY:
                error_message = f"Whisper CLI failed: {e.stderr if e.stderr else 'Unknown error'}"
                logging.error(f"Whisper CLI failed: {e.stderr if e.stderr else 'Unknown error'}")
        return JSONResponse(status_code=500, content={"error": error_message})
    except Exception:
        logging.exception(f"Unexpected error during transcription")
        return JSONResponse(
            status_code=500,
            content={"error": f"An unexpected error occurred: {str(e)}"}
        )
    finally:
        if input_wav_path and os.path.exists(input_wav_path):
            os.remove(input_wav_path)
        if original_temp_file_path and os.path.exists(
            original_temp_file_path
        ):  # Clean up original temp file
            os.remove(original_temp_file_path)
        if os.path.exists(f"{output_prefix}.txt"):
            os.remove(f"{output_prefix}.txt")

    return {"text": transcription.strip()}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "4444"))
    uvicorn.run(app, host="0.0.0.0", port=port)

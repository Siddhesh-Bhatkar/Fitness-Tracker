import subprocess
import tempfile
import os
def convert_opus_to_wav(opus_data: bytes) -> bytes:
    with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as webm_file:
        webm_file.write(opus_data)
        webm_path = webm_file.name
    
    wav_path = webm_path + ".wav"
    
    try:
        # Convert WebM/Opus to WAV using FFmpeg
        command = [
            'ffmpeg',
            '-y',  # Overwrite output file if it exists
            '-i', webm_path,
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            wav_path
        ]
        
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        with open(wav_path, 'rb') as wav_file:
            return wav_file.read()
            
    except subprocess.CalledProcessError as e:
        print("FFmpeg error:", e.stderr.decode())
        raise RuntimeError(f"FFmpeg conversion failed: {e.stderr.decode()}")
        
    finally:
        # Clean up temporary files
        for path in [webm_path, wav_path]:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
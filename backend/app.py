from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import logging
from src.music_generator import MusicGenerator
from src.tamil_processor import TamilProcessor
from src.instruments import InstrumentSynthesizer
import tempfile
import uuid
import json
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize components
music_generator = MusicGenerator()
tamil_processor = TamilProcessor()
instrument_synthesizer = InstrumentSynthesizer()

# Storage for generated songs (in production, use a database)
generated_songs = {}

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy", 
        "message": "Tamil AI Song Generator is running",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    })

@app.route('/generate-song', methods=['POST'])
def generate_song():
    try:
        data = request.json
        logger.info(f"Received generation request: {data}")
        
        # Extract parameters
        kavithai = data.get('kavithai', '')
        genre = data.get('genre', 'classical')
        mood = data.get('mood', 'peaceful')
        instruments = data.get('instruments', ['tabla', 'veena', 'flute'])
        voice_type = data.get('voice', 'male')
        duration = int(data.get('duration', 60))
        model = data.get('model', 'musicgen')
        
        # Validation
        if not kavithai.strip():
            return jsonify({"error": "Kavithai is required"}), 400
        
        if duration > 300:  # 5 minutes max
            return jsonify({"error": "Duration cannot exceed 5 minutes"}), 400
        
        if len(instruments) == 0:
            return jsonify({"error": "At least one instrument must be selected"}), 400
        
        # Process Tamil text
        logger.info("Processing Tamil kavithai...")
        processed_text = tamil_processor.process_kavithai(kavithai)
        
        # Generate music prompt
        music_prompt = tamil_processor.create_music_prompt(
            processed_text, genre, mood, instruments
        )
        
        # Generate base music
        logger.info(f"Generating music with {model}...")
        audio_path = music_generator.generate_music(
            prompt=music_prompt,
            duration=duration,
            model=model
        )
        
        # Add Tamil vocals if requested
        if voice_type != 'none':
            logger.info("Adding Tamil vocals...")
            audio_path = tamil_processor.add_tamil_vocals(
                audio_path, kavithai, voice_type
            )
        
        # Enhance with traditional instruments
        logger.info("Adding traditional instruments...")
        final_audio = instrument_synthesizer.enhance_with_instruments(
            audio_path, instruments, genre, mood
        )
        
        # Generate unique filename
        song_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"tamil_song_{timestamp}_{song_id[:8]}.wav"
        
        # Store song metadata
        generated_songs[song_id] = {
            "filename": output_filename,
            "audio_path": final_audio,
            "metadata": {
                "kavithai": kavithai[:200] + "..." if len(kavithai) > 200 else kavithai,
                "genre": genre,
                "mood": mood,
                "instruments": instruments,
                "voice": voice_type,
                "duration": duration,
                "model": model,
                "processed_text": processed_text,
                "music_prompt": music_prompt
            },
            "created_at": datetime.now().isoformat()
        }
        
        logger.info(f"Song generated successfully: {song_id}")
        
        return jsonify({
            "success": True,
            "song_id": song_id,
            "filename": output_filename,
            "download_url": f"/download/{song_id}",
            "stream_url": f"/stream/{song_id}",
            "metadata": generated_songs[song_id]["metadata"]
        })
        
    except Exception as e:
        logger.error(f"Error generating song: {str(e)}", exc_info=True)
        return jsonify({"error": f"Generation failed: {str(e)}"}), 500

@app.route('/download/<song_id>', methods=['GET'])
def download_song(song_id):
    try:
        if song_id not in generated_songs:
            return jsonify({"error": "Song not found"}), 404
        
        song_data = generated_songs[song_id]
        audio_file = song_data["audio_path"]
        
        if os.path.exists(audio_file):
            return send_file(
                audio_file, 
                as_attachment=True, 
                download_name=song_data["filename"],
                mimetype='audio/wav'
            )
        else:
            return jsonify({"error": "Audio file not found"}), 404
            
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/stream/<song_id>', methods=['GET'])
def stream_song(song_id):
    try:
        if song_id not in generated_songs:
            return jsonify({"error": "Song not found"}), 404
        
        song_data = generated_songs[song_id]
        audio_file = song_data["audio_path"]
        
        if os.path.exists(audio_file):
            return send_file(audio_file, mimetype='audio/wav')
        else:
            return jsonify({"error": "Audio file not found"}), 404
            
    except Exception as e:
        logger.error(f"Stream error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/models', methods=['GET'])
def get_available_models():
    return jsonify({
        "models": [
            {
                "id": "musicgen",
                "name": "MusicGen (Meta)",
                "description": "Advanced music generation with traditional instruments",
                "features": ["Open Source", "High Quality", "Multi-instrument"],
                "max_duration": 300,
                "supported_genres": ["classical", "carnatic", "folk", "devotional", "contemporary", "fusion"]
            },
            {
                "id": "audioldm",
                "name": "AudioLDM",
                "description": "Atmospheric background music generation",
                "features": ["Open Source", "Atmospheric", "Fast"],
                "max_duration": 180,
                "supported_genres": ["classical", "devotional", "contemporary"]
            },
            {
                "id": "musiclm",
                "name": "MusicLM Style",
                "description": "Cultural authenticity focused generation",
                "features": ["Cultural", "Authentic", "Detailed"],
                "max_duration": 240,
                "supported_genres": ["classical", "carnatic", "folk", "devotional"]
            }
        ]
    })

@app.route('/instruments', methods=['GET'])
def get_available_instruments():
    return jsonify({
        "instruments": instrument_synthesizer.get_instrument_list()
    })

@app.route('/songs', methods=['GET'])
def list_songs():
    """List all generated songs (for demo purposes)"""
    try:
        song_list = []
        for song_id, data in generated_songs.items():
            song_list.append({
                "song_id": song_id,
                "filename": data["filename"],
                "created_at": data["created_at"],
                "metadata": {
                    "genre": data["metadata"]["genre"],
                    "mood": data["metadata"]["mood"],
                    "duration": data["metadata"]["duration"],
                    "model": data["metadata"]["model"]
                }
            })
        
        return jsonify({"songs": song_list})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get generation statistics"""
    try:
        total_songs = len(generated_songs)
        models_used = {}
        genres_used = {}
        
        for data in generated_songs.values():
            model = data["metadata"]["model"]
            genre = data["metadata"]["genre"]
            
            models_used[model] = models_used.get(model, 0) + 1
            genres_used[genre] = genres_used.get(genre, 0) + 1
        
        return jsonify({
            "total_songs": total_songs,
            "models_used": models_used,
            "genres_used": genres_used,
            "server_uptime": "Running"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)

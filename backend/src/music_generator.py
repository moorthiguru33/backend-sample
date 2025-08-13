import torch
import torchaudio
import numpy as np
import librosa
import soundfile as sf
import tempfile
import os
import logging
from typing import Optional, Dict, List
import warnings
warnings.filterwarnings("ignore")

# Try to import transformers models
try:
    from transformers import AutoProcessor, MusicgenForConditionalGeneration
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: Transformers not available. Using fallback music generation.")

class MusicGenerator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {}
        self.sample_rate = 32000  # Standard for MusicGen
        self.logger = logging.getLogger(__name__)
        self.load_models()
    
    def load_models(self):
        """Load available music generation models"""
        try:
            if TRANSFORMERS_AVAILABLE:
                self.logger.info("Loading MusicGen model...")
                self.models['musicgen'] = {
                    'processor': AutoProcessor.from_pretrained("facebook/musicgen-medium"),
                    'model': MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-medium")
                }
                self.models['musicgen']['model'].to(self.device)
                self.logger.info("MusicGen model loaded successfully")
            else:
                self.logger.warning("MusicGen not available, using fallback")
                
        except Exception as e:
            self.logger.error(f"Error loading MusicGen: {e}")
        
        # Always have fallback available
        self.models['basic'] = True
        self.logger.info("Basic synthesis model loaded")
    
    def generate_music(self, prompt: str, duration: int = 60, model: str = "musicgen") -> str:
        """Generate music from text prompt"""
        try:
            self.logger.info(f"Generating music: model={model}, duration={duration}s")
            
            if model == "musicgen" and "musicgen" in self.models:
                return self._generate_with_musicgen(prompt, duration)
            elif model == "audioldm":
                return self._generate_with_audioldm(prompt, duration)
            elif model == "musiclm":
                return self._generate_with_musiclm(prompt, duration)
            else:
                return self._generate_basic_music(prompt, duration)
                
        except Exception as e:
            self.logger.error(f"Error in music generation: {e}")
            return self._generate_basic_music(prompt, duration)
    
    def _generate_with_musicgen(self, prompt: str, duration: int) -> str:
        """Generate music using MusicGen"""
        try:
            processor = self.models['musicgen']['processor']
            model = self.models['musicgen']['model']
            
            # Enhanced prompt for Tamil music
            enhanced_prompt = self._enhance_prompt_for_tamil(prompt)
            
            inputs = processor(
                text=[enhanced_prompt],
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            
            # Calculate tokens for duration (roughly 50 tokens per second)
            max_new_tokens = min(duration * 50, 1500)  # Cap for memory
            
            self.logger.info(f"Generating with prompt: {enhanced_prompt}")
            
            with torch.no_grad():
                audio_values = model.generate(
                    **inputs, 
                    max_new_tokens=max_new_tokens, 
                    do_sample=True, 
                    guidance_scale=3.0,
                    temperature=0.9
                )
            
            # Convert to numpy and save
            audio_np = audio_values[0, 0].cpu().numpy()
            
            # Ensure correct duration
            target_samples = int(duration * self.sample_rate)
            if len(audio_np) > target_samples:
                audio_np = audio_np[:target_samples]
            elif len(audio_np) < target_samples:
                # Pad with silence
                padding = target_samples - len(audio_np)
                audio_np = np.pad(audio_np, (0, padding), mode='constant')
            
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            sf.write(temp_file.name, audio_np, self.sample_rate)
            
            self.logger.info(f"MusicGen generation complete: {temp_file.name}")
            return temp_file.name
            
        except Exception as e:
            self.logger.error(f"MusicGen generation failed: {e}")
            return self._generate_basic_music(prompt, duration)
    
    def _generate_with_audioldm(self, prompt: str, duration: int) -> str:
        """Generate music using AudioLDM (placeholder - would need actual implementation)"""
        self.logger.info("AudioLDM not implemented, falling back to basic generation")
        return self._generate_basic_music(prompt, duration)
    
    def _generate_with_musiclm(self, prompt: str, duration: int) -> str:
        """Generate music using MusicLM style (placeholder)"""
        self.logger.info("MusicLM not implemented, falling back to basic generation")
        return self._generate_basic_music(prompt, duration)
    
    def _enhance_prompt_for_tamil(self, prompt: str) -> str:
        """Enhance prompt specifically for Tamil music generation"""
        enhancements = [
            "Traditional South Indian classical music",
            "Carnatic raga based composition",
            "Tamil cultural authentic sound",
            "Melodic and rhythmic patterns",
            "Professional recording quality"
        ]
        
        enhanced = f"{prompt}, {', '.join(enhancements)}"
        return enhanced[:500]  # Limit prompt length
    
    def _generate_basic_music(self, prompt: str, duration: int) -> str:
        """Fallback basic music generation using synthesis"""
        self.logger.info("Using basic music synthesis")
        
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Extract mood and genre from prompt for better synthesis
        mood = self._extract_mood_from_prompt(prompt)
        genre = self._extract_genre_from_prompt(prompt)
        
        # Generate music based on extracted parameters
        audio = self._synthesize_tamil_music(t, mood, genre, sample_rate)
        
        # Normalize
        audio = audio / np.max(np.abs(audio)) * 0.8
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        sf.write(temp_file.name, audio, sample_rate)
        
        self.logger.info(f"Basic synthesis complete: {temp_file.name}")
        return temp_file.name
    
    def _extract_mood_from_prompt(self, prompt: str) -> str:
        """Extract mood from prompt text"""
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ['peaceful', 'calm', 'serene', 'shankarabharanam']):
            return 'peaceful'
        elif any(word in prompt_lower for word in ['joyful', 'happy', 'celebratory', 'kalyani']):
            return 'joyful'
        elif any(word in prompt_lower for word in ['romantic', 'love', 'tender', 'mohanam']):
            return 'romantic'
        elif any(word in prompt_lower for word in ['devotional', 'spiritual', 'divine', 'bhairavi']):
            return 'devotional'
        elif any(word in prompt_lower for word in ['energetic', 'vigorous', 'lively']):
            return 'energetic'
        else:
            return 'peaceful'  # Default
    
    def _extract_genre_from_prompt(self, prompt: str) -> str:
        """Extract genre from prompt text"""
        prompt_lower = prompt.lower()
        
        if 'classical' in prompt_lower or 'carnatic' in prompt_lower:
            return 'classical'
        elif 'folk' in prompt_lower:
            return 'folk'
        elif 'devotional' in prompt_lower:
            return 'devotional'
        elif 'contemporary' in prompt_lower:
            return 'contemporary'
        else:
            return 'classical'  # Default
    
    def _synthesize_tamil_music(self, t: np.ndarray, mood: str, genre: str, sample_rate: int) -> np.ndarray:
        """Synthesize Tamil music based on parameters"""
        
        # Define raga scales
        raga_scales = {
            'peaceful': [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88],  # Shankarabharanam
            'joyful': [261.63, 293.66, 329.63, 369.99, 392.00, 440.00, 493.88],    # Kalyani
            'romantic': [261.63, 293.66, 329.63, 392.00, 440.00],                  # Mohanam
            'devotional': [261.63, 277.18, 311.13, 349.23, 392.00, 415.30, 466.16], # Bhairavi
            'energetic': [261.63, 277.18, 329.63, 349.23, 392.00, 415.30, 493.88]   # Kharaharapriya
        }
        
        scale = raga_scales.get(mood, raga_scales['peaceful'])
        
        # Generate melody
        melody = self._generate_melody(t, scale, mood)
        
        # Generate rhythm
        rhythm = self._generate_rhythm(t, genre, sample_rate)
        
        # Generate drone (tanpura effect)
        drone = self._generate_drone(t, scale[0])  # Root note drone
        
        # Mix components
        audio = melody * 0.6 + rhythm * 0.3 + drone * 0.1
        
        # Add some reverb effect
        audio = self._apply_reverb(audio, sample_rate)
        
        return audio
    
    def _generate_melody(self, t: np.ndarray, scale: List[float], mood: str) -> np.ndarray:
        """Generate melodic content"""
        melody = np.zeros_like(t)
        
        # Determine note duration based on mood
        note_duration = 0.5 if mood == 'energetic' else 1.0
        samples_per_note = int(note_duration * len(t) / t[-1])
        
        for i in range(0, len(t), samples_per_note):
            end_idx = min(i + samples_per_note, len(t))
            note_t = t[i:end_idx] - t[i]
            
            # Choose frequency from scale
            freq = scale[i // samples_per_note % len(scale)]
            
            # Add harmonics for richer sound
            note = np.zeros_like(note_t)
            harmonics = [1.0, 0.5, 0.3, 0.2, 0.1]
            
            for h, amp in enumerate(harmonics):
                note += amp * np.sin(2 * np.pi * freq * (h + 1) * note_t)
            
            # Apply envelope
            envelope = np.exp(-note_t * 2)  # Exponential decay
            note *= envelope
            
            melody[i:end_idx] = note
        
        return melody
    
    def _generate_rhythm(self, t: np.ndarray, genre: str, sample_rate: int) -> np.ndarray:
        """Generate rhythmic patterns"""
        rhythm = np.zeros_like(t)
        
        # Define rhythm patterns
        if genre == 'classical':
            beat_pattern = [1, 0, 1, 0, 1, 0, 1, 0]  # 4/4 classical
            bpm = 90
        elif genre == 'folk':
            beat_pattern = [1, 0, 1, 1, 0, 1]  # 6/8 folk
            bpm = 120
        elif genre == 'devotional':
            beat_pattern = [1, 0, 0, 1, 0, 0]  # 3/4 devotional
            bpm = 80
        else:
            beat_pattern = [1, 0, 1, 0, 1, 0, 1, 0]
            bpm = 100
        
        beat_duration = 60 / bpm
        
        for i, strength in enumerate(beat_pattern * int(len(t) / sample_rate / beat_duration / len(beat_pattern) + 1)):
            beat_time = i * beat_duration
            beat_sample = int(beat_time * sample_rate)
            
            if beat_sample < len(rhythm) and strength > 0:
                # Generate tabla-like sound
                hit_duration = 0.1
                hit_samples = int(hit_duration * sample_rate)
                
                if beat_sample + hit_samples < len(rhythm):
                    hit_t = np.linspace(0, hit_duration, hit_samples)
                    
                    # Low frequency for bass drum effect
                    freq = 60 + np.random.uniform(-10, 10)
                    hit = np.sin(2 * np.pi * freq * hit_t) * np.exp(-hit_t * 20)
                    
                    rhythm[beat_sample:beat_sample + hit_samples] += hit * strength * 0.5
        
        return rhythm
    
    def _generate_drone(self, t: np.ndarray, root_freq: float) -> np.ndarray:
        """Generate tanpura-like drone"""
        drone = np.zeros_like(t)
        
        # Root and fifth
        freqs = [root_freq, root_freq * 1.5]
        
        for freq in freqs:
            drone += np.sin(2 * np.pi * freq * t) / len(freqs)
        
        # Add slight modulation for realism
        modulation = 1 + 0.02 * np.sin(2 * np.pi * 0.5 * t)
        drone *= modulation
        
        return drone * 0.3  # Keep drone subtle
    
    def _apply_reverb(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply simple reverb effect"""
        # Simple delay-based reverb
        delay_samples = int(0.1 * sample_rate)  # 100ms delay
        
        if len(audio) > delay_samples:
            delayed = np.zeros_like(audio)
            delayed[delay_samples:] = audio[:-delay_samples] * 0.3
            audio = audio + delayed
        
        return audio

    def get_supported_models(self) -> List[str]:
        """Get list of supported models"""
        return list(self.models.keys())

import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from scipy.io import wavfile
import tempfile
from typing import List, Dict, Tuple
import logging

class InstrumentSynthesizer:
    def __init__(self):
        self.instrument_profiles = self._load_instrument_profiles()
        self.sample_rate = 44100
        self.logger = logging.getLogger(__name__)
    
    def _load_instrument_profiles(self) -> Dict:
        """Load comprehensive instrument synthesis profiles"""
        return {
            'tabla': {
                'type': 'percussion',
                'frequency_range': [50, 200],
                'attack': 0.01,
                'decay': 0.3,
                'sustain': 0.1,
                'release': 0.2,
                'harmonics': [1, 0.3, 0.1, 0.05],
                'character': 'rhythmic_bass',
                'cultural_weight': 1.0,
                'traditional_patterns': ['terekite', 'dha_dha', 'ti_ti']
            },
            'mridangam': {
                'type': 'percussion',
                'frequency_range': [60, 250],
                'attack': 0.005,
                'decay': 0.4,
                'sustain': 0.05,
                'release': 0.3,
                'harmonics': [1, 0.4, 0.2, 0.1, 0.05],
                'character': 'complex_rhythmic',
                'cultural_weight': 1.0,
                'traditional_patterns': ['tha_ki_ta', 'tha_tha', 'ki_ta_tha_ka']
            },
            'veena': {
                'type': 'string',
                'frequency_range': [196, 1568],  # G3 to G6
                'attack': 0.1,
                'decay': 0.3,
                'sustain': 0.6,
                'release': 0.5,
                'harmonics': [1, 0.5, 0.3, 0.2, 0.1, 0.05],
                'character': 'melodic_lead',
                'cultural_weight': 1.0,
                'ornaments': ['gamaka', 'meend', 'andolan']
            },
            'flute': {
                'type': 'wind',
                'frequency_range': [261, 2093],  # C4 to C7
                'attack': 0.05,
                'decay': 0.1,
                'sustain': 0.8,
                'release': 0.3,
                'harmonics': [1, 0.3, 0.1, 0.05, 0.02],
                'character': 'melodic_flowing',
                'cultural_weight': 0.9,
                'breath_patterns': ['legato', 'staccato', 'vibrato']
            },
            'violin': {
                'type': 'string',
                'frequency_range': [196, 3136],  # G3 to G7
                'attack': 0.05,
                'decay': 0.2,
                'sustain': 0.7,
                'release': 0.4,
                'harmonics': [1, 0.6, 0.4, 0.3, 0.2, 0.1],
                'character': 'expressive_harmony',
                'cultural_weight': 0.8,
                'bowing_styles': ['legato', 'staccato', 'tremolo']
            },
            'keyboard': {
                'type': 'keyboard',
                'frequency_range': [65, 2093],  # C2 to C7
                'attack': 0.01,
                'decay': 0.2,
                'sustain': 0.5,
                'release': 0.3,
                'harmonics': [1, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05],
                'character': 'harmonic_foundation',
                'cultural_weight': 0.7,
                'voices': ['harmonium', 'organ', 'piano']
            },
            'thavil': {
                'type': 'percussion',
                'frequency_range': [80, 300],
                'attack': 0.005,
                'decay': 0.2,
                'sustain': 0.1,
                'release': 0.4,
                'harmonics': [1, 0.5, 0.3, 0.2, 0.1],
                'character': 'powerful_rhythmic',
                'cultural_weight': 1.0,
                'traditional_patterns': ['tha_dhimi', 'tha_ka_dha', 'dhimi_dhimi']
            },
            'nadhaswaram': {
                'type': 'wind',
                'frequency_range': [174, 1397],  # F3 to F6
                'attack': 0.1,
                'decay': 0.2,
                'sustain': 0.7,
                'release': 0.4,
                'harmonics': [1, 0.8, 0.6, 0.4, 0.2, 0.1],
                'character': 'majestic_lead',
                'cultural_weight': 1.0,
                'ornaments': ['heavy_vibrato', 'glissando', 'trill']
            }
        }
    
    def enhance_with_instruments(self, audio_path: str, instruments: List[str], 
                                genre: str, mood: str) -> str:
        """Enhance audio with traditional Tamil instruments"""
        try:
            self.logger.info(f"Enhancing audio with instruments: {instruments}")
            
            # Load original audio
            original_audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Analyze original audio for better integration
            audio_analysis = self._analyze_audio(original_audio, sr)
            
            # Create instrument tracks
            instrument_tracks = []
            
            for instrument in instruments:
                if instrument in self.instrument_profiles:
                    self.logger.info(f"Synthesizing {instrument}")
                    track = self._synthesize_instrument(
                        instrument, len(original_audio), genre, mood, audio_analysis
                    )
                    instrument_tracks.append((instrument, track))
            
            # Mix all tracks with intelligent balancing
            enhanced_audio = self._intelligent_mix(
                original_audio, instrument_tracks, genre, mood
            )
            
            # Apply final processing
            enhanced_audio = self._apply_final_processing(enhanced_audio, genre)
            
            # Save enhanced audio
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            sf.write(temp_file.name, enhanced_audio, self.sample_rate)
            
            self.logger.info(f"Enhancement complete: {temp_file.name}")
            return temp_file.name
            
        except Exception as e:
            self.logger.error(f"Error enhancing with instruments: {e}")
            return audio_path
    
    def _analyze_audio(self, audio: np.ndarray, sr: int) -> Dict:
        """Analyze audio characteristics for better instrument integration"""
        
        # Basic audio analysis
        rms_energy = np.sqrt(np.mean(audio**2))
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
        
        # Frequency analysis
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        
        # Dominant frequency bins
        freq_bins = librosa.fft_frequencies(sr=sr)
        dominant_freqs = freq_bins[np.argmax(magnitude, axis=0)]
        
        return {
            'rms_energy': rms_energy,
            'spectral_centroid': np.mean(spectral_centroid),
            'tempo': tempo,
            'beats': beats,
            'dominant_frequencies': dominant_freqs,
            'duration': len(audio) / sr,
            'dynamic_range': np.max(audio) - np.min(audio)
        }
    
    def _synthesize_instrument(self, instrument: str, audio_length: int, 
                              genre: str, mood: str, audio_analysis: Dict) -> np.ndarray:
        """Synthesize a specific instrument track with cultural authenticity"""
        profile = self.instrument_profiles[instrument]
        
        # Create time array
        t = np.linspace(0, audio_length / self.sample_rate, audio_length)
        
        if profile['type'] == 'percussion':
            return self._synthesize_percussion_advanced(
                instrument, t, genre, mood, audio_analysis
            )
        elif profile['type'] == 'string':
            return self._synthesize_string_advanced(
                instrument, t, genre, mood, audio_analysis
            )
        elif profile['type'] == 'wind':
            return self._synthesize_wind_advanced(
                instrument, t, genre, mood, audio_analysis
            )
        elif profile['type'] == 'keyboard':
            return self._synthesize_keyboard_advanced(
                instrument, t, genre, mood, audio_analysis
            )
        else:
            return np.zeros_like(t)
    
    def _synthesize_percussion_advanced(self, instrument: str, t: np.ndarray, 
                                      genre: str, mood: str, audio_analysis: Dict) -> np.ndarray:
        """Advanced percussion synthesis with traditional patterns"""
        profile = self.instrument_profiles[instrument]
        
        # Get tempo from audio analysis or set default
        tempo = audio_analysis.get('tempo', self._get_default_tempo(mood))
        
        # Determine rhythm pattern based on genre and instrument
        pattern = self._get_rhythm_pattern(instrument, genre, mood)
        
        # Generate percussion track
        percussion = np.zeros_like(t)
        beat_duration = 60 / tempo  # seconds per beat
        
        pattern_length = len(pattern)
        total_beats = int(len(t) / self.sample_rate / beat_duration)
        
        for beat_idx in range(total_beats):
            pattern_idx = beat_idx % pattern_length
            beat_strength = pattern[pattern_idx]
            
            if beat_strength > 0:
                beat_time = beat_idx * beat_duration
                beat_sample = int(beat_time * self.sample_rate)
                
                if beat_sample < len(percussion):
                    # Generate drum hit with appropriate characteristics
                    hit = self._generate_drum_hit(
                        instrument, beat_strength, profile
                    )
                    
                    # Apply hit to track
                    end_sample = min(beat_sample + len(hit), len(percussion))
                    hit_length = end_sample - beat_sample
                    percussion[beat_sample:end_sample] += hit[:hit_length]
        
        return percussion
    
    def _synthesize_string_advanced(self, instrument: str, t: np.ndarray, 
                                  genre: str, mood: str, audio_analysis: Dict) -> np.ndarray:
        """Advanced string synthesis with traditional ornaments"""
        profile = self.instrument_profiles[instrument]
        
        # Define scale and melodic patterns
        scale_freqs = self._get_raga_scale(mood)
        
        # Adapt frequencies to instrument range
        scale_freqs = self._adapt_to_instrument_range(scale_freqs, profile)
        
        # Generate melodic content
        string_audio = np.zeros_like(t)
        
        # Determine note timing based on genre and mood
        note_duration = self._get_note_duration(genre, mood)
        
        notes_count = int(len(t) / self.sample_rate / note_duration)
        
        for note_idx in range(notes_count):
            note_start = note_idx * note_duration
            note_end = (note_idx + 1) * note_duration
            
            start_sample = int(note_start * self.sample_rate)
            end_sample = int(note_end * self.sample_rate)
            
            if end_sample > len(string_audio):
                end_sample = len(string_audio)
            
            if start_sample >= end_sample:
                continue
            
            # Choose frequency from scale
            freq = scale_freqs[note_idx % len(scale_freqs)]
            
            # Generate note with ornaments
            note_length = end_sample - start_sample
            note = self._generate_string_note(
                freq, note_length, profile, instrument
            )
            
            string_audio[start_sample:end_sample] += note
        
        return string_audio
    
    def _synthesize_wind_advanced(self, instrument: str, t: np.ndarray, 
                                genre: str, mood: str, audio_analysis: Dict) -> np.ndarray:
        """Advanced wind instrument synthesis"""
        profile = self.instrument_profiles[instrument]
        
        # Wind instruments often play melodic phrases
        phrase_length = 2.0  # seconds per phrase
        phrases_count = int(len(t) / self.sample_rate / phrase_length)
        
        wind_audio = np.zeros_like(t)
        scale_freqs = self._get_raga_scale(mood)
        scale_freqs = self._adapt_to_instrument_range(scale_freqs, profile)
        
        for phrase_idx in range(phrases_count):
            phrase_start = phrase_idx * phrase_length
            phrase_end = (phrase_idx + 1) * phrase_length
            
            start_sample = int(phrase_start * self.sample_rate)
            end_sample = int(phrase_end * self.sample_rate)
            
            if end_sample > len(wind_audio):
                end_sample = len(wind_audio)
            
            # Generate melodic phrase
            phrase = self._generate_wind_phrase(
                scale_freqs, end_sample - start_sample, profile, instrument
            )
            
            wind_audio[start_sample:end_sample] += phrase
        
        return wind_audio
    
    def _synthesize_keyboard_advanced(self, instrument: str, t: np.ndarray, 
                                    genre: str, mood: str, audio_analysis: Dict) -> np.ndarray:
        """Advanced keyboard synthesis with harmonic support"""
        profile = self.instrument_profiles[instrument]
        
        # Keyboard provides harmonic foundation
        chord_duration = 2.0  # seconds per chord
        chords_count = int(len(t) / self.sample_rate / chord_duration)
        
        keyboard_audio = np.zeros_like(t)
        chord_progression = self._get_chord_progression(mood)
        
        for chord_idx in range(chords_count):
            chord_start = chord_idx * chord_duration
            chord_end = (chord_idx + 1) * chord_duration
            
            start_sample = int(chord_start * self.sample_rate)
            end_sample = int(chord_end * self.sample_rate)
            
            if end_sample > len(keyboard_audio):
                end_sample = len(keyboard_audio)
            
            # Get chord frequencies
            chord_freqs = chord_progression[chord_idx % len(chord_progression)]
            
            # Generate chord
            chord = self._generate_keyboard_chord(
                chord_freqs, end_sample - start_sample, profile
            )
            
            keyboard_audio[start_sample:end_sample] += chord
        
        return keyboard_audio
    
    def _generate_drum_hit(self, instrument: str, strength: float, 
                          profile: Dict) -> np.ndarray:
        """Generate authentic drum hit sound"""
        
        hit_duration = 0.1  # 100ms hit
        hit_samples = int(hit_duration * self.sample_rate)
        hit_t = np.linspace(0, hit_duration, hit_samples)
        
        # Base frequency for this instrument
        base_freq = np.random.uniform(
            profile['frequency_range'][0], 
            profile['frequency_range'][1]
        )
        
        # Generate hit sound with harmonics
        hit_sound = np.zeros_like(hit_t)
        
        for i, harmonic_strength in enumerate(profile['harmonics']):
            harmonic_freq = base_freq * (i + 1)
            harmonic_wave = np.sin(2 * np.pi * harmonic_freq * hit_t)
            hit_sound += harmonic_strength * harmonic_wave
        
        # Apply ADSR envelope
        envelope = self._create_adsr_envelope(hit_samples, profile)
        hit_sound *= envelope * strength
        
        # Add noise component for realism
        if instrument in ['tabla', 'mridangam', 'thavil']:
            noise = np.random.normal(0, 0.1, len(hit_sound))
            hit_sound += noise * 0.1
        
        return hit_sound * 0.3
    
    def _generate_string_note(self, freq: float, note_length: int, 
                             profile: Dict, instrument: str) -> np.ndarray:
        """Generate string note with traditional ornaments"""
        
        note_t = np.linspace(0, note_length / self.sample_rate, note_length)
        
        # Base note
        note = np.zeros_like(note_t)
        
        # Add harmonics
        for i, harmonic_strength in enumerate(profile['harmonics']):
            harmonic_freq = freq * (i + 1)
            harmonic_wave = np.sin(2 * np.pi * harmonic_freq * note_t)
            note += harmonic_strength * harmonic_wave
        
        # Add traditional ornaments
        if instrument == 'veena':
            # Add gamaka (oscillations)
            gamaka = 1 + 0.05 * np.sin(2 * np.pi * 5 * note_t)
            note *= gamaka
        
        # Apply envelope
        envelope = self._create_adsr_envelope(note_length, profile)
        note *= envelope
        
        return note * 0.2
    
    def _generate_wind_phrase(self, scale_freqs: List[float], phrase_length: int, 
                             profile: Dict, instrument: str) -> np.ndarray:
        """Generate wind instrument phrase"""
        
        phrase_t = np.linspace(0, phrase_length / self.sample_rate, phrase_length)
        phrase = np.zeros_like(phrase_t)
        
        # Number of notes in phrase
        notes_in_phrase = 4
        note_duration = len(phrase_t) / notes_in_phrase
        
        for note_idx in range(notes_in_phrase):
            start_idx = int(note_idx * note_duration)
            end_idx = int((note_idx + 1) * note_duration)
            
            freq = scale_freqs[note_idx % len(scale_freqs)]
            note_t = phrase_t[start_idx:end_idx] - phrase_t[start_idx]
            
            # Generate note
            note = np.sin(2 * np.pi * freq * note_t)
            
            # Add harmonics
            for i, harmonic_strength in enumerate(profile['harmonics'][1:], 2):
                note += harmonic_strength * np.sin(2 * np.pi * freq * i * note_t)
            
            # Add breath characteristics
            if instrument == 'flute':
                breath_modulation = 1 + 0.02 * np.sin(2 * np.pi * 3 * note_t)
                note *= breath_modulation
            
            # Apply note envelope
            note_envelope = np.exp(-note_t) * (1 - np.exp(-note_t * 5))
            note *= note_envelope
            
            phrase[start_idx:end_idx] = note
        
        return phrase * 0.15
    
    def _generate_keyboard_chord(self, chord_freqs: List[float], chord_length: int, 
                                profile: Dict) -> np.ndarray:
        """Generate keyboard chord"""
        
        chord_t = np.linspace(0, chord_length / self.sample_rate, chord_length)
        chord = np.zeros_like(chord_t)
        
        for freq in chord_freqs:
            # Generate chord tone
            tone = np.sin(2 * np.pi * freq * chord_t)
            
            # Add harmonics
            for i, harmonic_strength in enumerate(profile['harmonics'][1:], 2):
                tone += harmonic_strength * np.sin(2 * np.pi * freq * i * chord_t)
            
            chord += tone / len(chord_freqs)
        
        # Apply envelope
        envelope = self._create_adsr_envelope(chord_length, profile)
        chord *= envelope
        
        return chord * 0.1
    
    def _intelligent_mix(self, original: np.ndarray, instrument_tracks: List[Tuple[str, np.ndarray]], 
                        genre: str, mood: str) -> np.ndarray:
        """Intelligently mix tracks based on cultural importance and audio characteristics"""
        
        # Start with original audio
        mixed = original * 0.4  # Reduce original to make room for instruments
        
        # Calculate mixing weights based on cultural importance and genre
        mixing_weights = self._calculate_mixing_weights(instrument_tracks, genre, mood)
        
        # Mix each instrument track
        for (instrument, track), weight in zip(instrument_tracks, mixing_weights):
            if len(track) == len(mixed):
                mixed += track * weight
        
        # Apply dynamic range compression
        mixed = self._apply_compression(mixed)
        
        # Normalize to prevent clipping
        max_amplitude = np.max(np.abs(mixed))
        if max_amplitude > 0.95:
            mixed = mixed * 0.95 / max_amplitude
        
        return mixed
    
    def _calculate_mixing_weights(self, instrument_tracks: List[Tuple[str, np.ndarray]], 
                                 genre: str, mood: str) -> List[float]:
        """Calculate mixing weights based on instrument importance"""
        
        weights = []
        
        for instrument, _ in instrument_tracks:
            profile = self.instrument_profiles[instrument]
            
            # Base weight from cultural importance
            weight = profile['cultural_weight'] * 0.3
            
            # Adjust based on genre
            if genre == 'classical' and instrument in ['veena', 'mridangam']:
                weight *= 1.5
            elif genre == 'folk' and instrument in ['tabla', 'flute']:
                weight *= 1.3
            elif genre == 'devotional' and instrument in ['nadhaswaram', 'thavil']:
                weight *= 1.4
            
            # Adjust based on mood
            if mood == 'peaceful' and instrument in ['flute', 'veena']:
                weight *= 1.2
            elif mood == 'energetic' and instrument in ['tabla', 'mridangam']:
                weight *= 1.3
            
            # Ensure weight is reasonable
            weight = max(0.1, min(weight, 0.5))
            weights.append(weight)
        
        return weights
    
    def _apply_compression(self, audio: np.ndarray, threshold: float = 0.7, 
                          ratio: float = 3.0) -> np.ndarray:
        """Apply dynamic range compression"""
        
        # Simple compression algorithm
        compressed = audio.copy()
        
        # Find samples above threshold
        above_threshold = np.abs(compressed) > threshold
        
        # Apply compression to samples above threshold
        compressed[above_threshold] = (
            np.sign(compressed[above_threshold]) * 
            (threshold + (np.abs(compressed[above_threshold]) - threshold) / ratio)
        )
        
        return compressed
    
    def _apply_final_processing(self, audio: np.ndarray, genre: str) -> np.ndarray:
        """Apply final processing effects"""
        
        # Apply gentle EQ based on genre
        if genre == 'classical':
            # Boost mid frequencies for classical warmth
            audio = self._apply_eq_boost(audio, 500, 2000, 1.1)
        elif genre == 'folk':
            # Boost higher frequencies for brightness
            audio = self._apply_eq_boost(audio, 1000, 4000, 1.15)
        
        # Apply subtle reverb
        audio = self._apply_reverb(audio)
        
        return audio
    
    def _apply_eq_boost(self, audio: np.ndarray, low_freq: float, 
                       high_freq: float, gain: float) -> np.ndarray:
        """Apply simple EQ boost in frequency range"""
        
        # Simple frequency domain boost (very basic implementation)
        fft = np.fft.fft(audio)
        freqs = np.fft.fftfreq(len(audio), 1/self.sample_rate)
        
        # Create boost mask
        boost_mask = (np.abs(freqs) >= low_freq) & (np.abs(freqs) <= high_freq)
        fft[boost_mask] *= gain
        
        # Convert back to time domain
        boosted_audio = np.real(np.fft.ifft(fft))
        
        return boosted_audio
    
    def _apply_reverb(self, audio: np.ndarray) -> np.ndarray:
        """Apply simple reverb effect"""
        
        # Simple delay-based reverb
        delay1_samples = int(0.03 * self.sample_rate)  # 30ms delay
        delay2_samples = int(0.07 * self.sample_rate)  # 70ms delay
        
        reverb_audio = audio.copy()
        
        if len(audio) > delay1_samples:
            reverb_audio[delay1_samples:] += audio[:-delay1_samples] * 0.3
        
        if len(audio) > delay2_samples:
            reverb_audio[delay2_samples:] += audio[:-delay2_samples] * 0.2
        
        return reverb_audio
    
    # Helper methods
    def _get_default_tempo(self, mood: str) -> float:
        """Get default tempo based on mood"""
        tempo_map = {
            'peaceful': 70,
            'joyful': 120,
            'romantic': 80,
            'devotional': 60,
            'energetic': 140,
            'melancholic': 65
        }
        return tempo_map.get(mood, 90)
    
    def _get_rhythm_pattern(self, instrument: str, genre: str, mood: str) -> List[float]:
        """Get traditional rhythm patterns for instruments"""
        
        patterns = {
            'tabla': {
                'classical': [1, 0, 0.5, 0, 1, 0, 0.5, 0],
                'folk': [1, 0.5, 1, 0.5, 1, 0.5],
                'devotional': [1, 0, 0, 1, 0, 0],
                'contemporary': [1, 0, 1, 0, 1, 0, 1, 0]
            },
            'mridangam': {
                'classical': [1, 0, 0.3, 0.7, 0, 0.5, 0.8, 0],
                'carnatic': [1, 0.3, 0.5, 0.3, 1, 0.5, 0.3, 0.8],
                'devotional': [1, 0, 0.5, 0, 0.8, 0, 0.3, 0]
            },
            'thavil': {
                'folk': [1, 0.8, 0, 1, 0.6, 0, 1, 0],
                'devotional': [1, 0, 0.9, 0, 1, 0.7, 0, 0.5],
                'classical': [1, 0.4, 0.6, 0.4, 1, 0.6, 0.4, 0.8]
            }
        }
        
        # Get pattern for instrument and genre
        instrument_patterns = patterns.get(instrument, {})
        pattern = instrument_patterns.get(genre, [1, 0, 1, 0])
        
        # Adjust pattern based on mood
        if mood == 'energetic':
            pattern = [p * 1.2 if p > 0 else p for p in pattern]
        elif mood == 'peaceful':
            pattern = [p * 0.8 if p > 0 else p for p in pattern]
        
        return pattern
    
    def _get_raga_scale(self, mood: str) -> List[float]:
        """Get raga scale frequencies based on mood"""
        scales = {
            'peaceful': [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88],  # Shankarabharanam
            'joyful': [261.63, 293.66, 329.63, 369.99, 392.00, 440.00, 493.88],    # Kalyani
            'romantic': [261.63, 293.66, 329.63, 392.00, 440.00],                  # Mohanam
            'devotional': [261.63, 277.18, 311.13, 349.23, 392.00, 415.30, 466.16], # Bhairavi
            'energetic': [261.63, 277.18, 329.63, 349.23, 392.00, 415.30, 493.88],   # Kharaharapriya
            'melancholic': [261.63, 277.18, 311.13, 349.23, 392.00, 415.30, 466.16]  # Sahana
        }
        return scales.get(mood, scales['peaceful'])
    
    def _adapt_to_instrument_range(self, frequencies: List[float], profile: Dict) -> List[float]:
        """Adapt frequencies to instrument range"""
        adapted = []
        for freq in frequencies:
            while freq < profile['frequency_range'][0]:
                freq *= 2
            while freq > profile['frequency_range'][1]:
                freq /= 2
            adapted.append(freq)
        return adapted
    
    def _get_note_duration(self, genre: str, mood: str) -> float:
        """Get note duration based on genre and mood"""
        durations = {
            'classical': 1.0,
            'folk': 0.8,
            'devotional': 1.2,
            'contemporary': 0.6
        }
        base_duration = durations.get(genre, 1.0)
        
        # Adjust for mood
        if mood == 'energetic':
            base_duration *= 0.8
        elif mood == 'peaceful':
            base_duration *= 1.2
            
        return base_duration
    
    def _get_chord_progression(self, mood: str) -> List[List[float]]:
        """Get chord progression for mood"""
        progressions = {
            'peaceful': [
                [261.63, 329.63, 392.00],  # C major
                [293.66, 369.99, 440.00],  # D major
                [349.23, 440.00, 523.25],  # F major
                [261.63, 329.63, 392.00]   # C major
            ],
            'joyful': [
                [261.63, 329.63, 392.00],  # C major
                [392.00, 493.88, 587.33],  # G major
                [349.23, 440.00, 523.25],  # F major
                [261.63, 329.63, 392.00]   # C major
            ],
            'romantic': [
                [261.63, 329.63, 392.00],  # C major
                [220.00, 277.18, 329.63],  # A minor
                [349.23, 440.00, 523.25],  # F major
                [392.00, 493.88, 587.33]   # G major
            ],
            'devotional': [
                [261.63, 311.13, 392.00],  # C minor-ish
                [293.66, 349.23, 440.00],  # D minor-ish
                [277.18, 329.63, 415.30],  # Bb major-ish
                [261.63, 311.13, 392.00]   # C minor-ish
            ]
        }
        return progressions.get(mood, progressions['peaceful'])
    
    def _create_adsr_envelope(self, length: int, profile: Dict) -> np.ndarray:
        """Create ADSR envelope for realistic sound"""
        envelope = np.ones(length)
        
        attack_samples = int(profile['attack'] * self.sample_rate)
        decay_samples = int(profile['decay'] * self.sample_rate)
        release_samples = int(profile['release'] * self.sample_rate)
        
        # Attack
        if attack_samples > 0 and attack_samples < length:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        
        # Decay
        if decay_samples > 0 and attack_samples + decay_samples < length:
            decay_end = attack_samples + decay_samples
            envelope[attack_samples:decay_end] = np.linspace(1, profile['sustain'], decay_samples)
        
        # Sustain (middle part remains at sustain level)
        sustain_start = attack_samples + decay_samples
        sustain_end = max(length - release_samples, sustain_start)
        if sustain_start < sustain_end:
            envelope[sustain_start:sustain_end] = profile['sustain']
        
        # Release
        if release_samples > 0 and release_samples < length:
            envelope[-release_samples:] = np.linspace(profile['sustain'], 0, release_samples)
        
        return envelope

    def get_instrument_list(self) -> List[Dict]:
        """Get list of available instruments"""
        return [
            {
                "id": key,
                "name": key.title(),
                "type": profile["type"],
                "cultural_weight": profile["cultural_weight"]
            }
            for key, profile in self.instrument_profiles.items()
        ]

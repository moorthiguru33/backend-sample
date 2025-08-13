import re
import numpy as np
import librosa
import soundfile as sf
from typing import List, Dict, Tuple
import tempfile
import os
import logging

class TamilProcessor:
    def __init__(self):
        self.tamil_phonemes = self._load_tamil_phonemes()
        self.raga_mappings = self._load_raga_mappings()
        self.mood_keywords = self._load_mood_keywords()
        self.logger = logging.getLogger(__name__)
    
    def _load_tamil_phonemes(self) -> Dict[str, str]:
        """Load Tamil phoneme mappings for pronunciation"""
        return {
            # Vowels
            'அ': 'a', 'ஆ': 'aa', 'இ': 'i', 'ஈ': 'ii', 'உ': 'u', 'ஊ': 'uu',
            'எ': 'e', 'ஏ': 'ee', 'ஐ': 'ai', 'ஒ': 'o', 'ஓ': 'oo', 'ஔ': 'au',
            
            # Consonants
            'க': 'ka', 'ங': 'nga', 'ச': 'cha', 'ஞ': 'nya', 'ட': 'ta',
            'ண': 'na', 'த': 'tha', 'ந': 'nha', 'ப': 'pa', 'ம': 'ma',
            'ய': 'ya', 'ர': 'ra', 'ல': 'la', 'வ': 'va', 'ழ': 'zha',
            'ள': 'lla', 'ற': 'rra', 'ன': 'nna',
            
            # Additional characters
            'ஸ': 'sa', 'ஷ': 'sha', 'ஹ': 'ha', 'க்ஷ': 'ksha',
            
            # Combined characters (basic combinations)
            'கா': 'kaa', 'கி': 'ki', 'கீ': 'kii', 'கு': 'ku', 'கூ': 'kuu',
            'கெ': 'ke', 'கே': 'kee', 'கை': 'kai', 'கொ': 'ko', 'கோ': 'koo',
            'தா': 'thaa', 'தி': 'thi', 'தீ': 'thii', 'து': 'thu', 'தூ': 'thuu',
            'மா': 'maa', 'மி': 'mi', 'மீ': 'mii', 'மு': 'mu', 'மூ': 'muu',
            'னா': 'naa', 'னி': 'ni', 'னீ': 'nii', 'னு': 'nu', 'னூ': 'nuu',
            'ரா': 'raa', 'ரி': 'ri', 'ரீ': 'rii', 'ரு': 'ru', 'ரூ': 'ruu',
            'லா': 'laa', 'லி': 'li', 'லீ': 'lii', 'லு': 'lu', 'லூ': 'luu',
            'வா': 'vaa', 'வி': 'vi', 'வீ': 'vii', 'வு': 'vu', 'வூ': 'vuu'
        }
    
    def _load_raga_mappings(self) -> Dict[str, Dict]:
        """Load Carnatic raga to mood mappings with musical properties"""
        return {
            'peaceful': {
                'raga': 'Shankarabharanam',
                'notes': ['Sa', 'Ri', 'Ga', 'Ma', 'Pa', 'Dha', 'Ni'],
                'frequencies': [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88],
                'characteristics': ['serene', 'balanced', 'complete', 'majestic'],
                'time_preference': 'morning',
                'tempo': 'medium'
            },
            'joyful': {
                'raga': 'Kalyani',
                'notes': ['Sa', 'Ri', 'Ga', 'Ma#', 'Pa', 'Dha', 'Ni'],
                'frequencies': [261.63, 293.66, 329.63, 369.99, 392.00, 440.00, 493.88],
                'characteristics': ['bright', 'uplifting', 'celebratory', 'optimistic'],
                'time_preference': 'afternoon',
                'tempo': 'fast'
            },
            'romantic': {
                'raga': 'Mohanam',
                'notes': ['Sa', 'Ri', 'Ga', 'Pa', 'Dha'],
                'frequencies': [261.63, 293.66, 329.63, 392.00, 440.00],
                'characteristics': ['beautiful', 'charming', 'pleasant', 'flowing'],
                'time_preference': 'evening',
                'tempo': 'medium'
            },
            'devotional': {
                'raga': 'Bhairavi',
                'notes': ['Sa', 'Ri♭', 'Ga♭', 'Ma', 'Pa', 'Dha♭', 'Ni♭'],
                'frequencies': [261.63, 277.18, 311.13, 349.23, 392.00, 415.30, 466.16],
                'characteristics': ['solemn', 'devotional', 'serious', 'introspective'],
                'time_preference': 'dawn',
                'tempo': 'slow'
            },
            'energetic': {
                'raga': 'Kharaharapriya',
                'notes': ['Sa', 'Ri♭', 'Ga', 'Ma', 'Pa', 'Dha♭', 'Ni'],
                'frequencies': [261.63, 277.18, 329.63, 349.23, 392.00, 415.30, 493.88],
                'characteristics': ['powerful', 'bold', 'dramatic', 'intense'],
                'time_preference': 'noon',
                'tempo': 'fast'
            },
            'melancholic': {
                'raga': 'Sahana',
                'notes': ['Sa', 'Ri♭', 'Ga♭', 'Ma', 'Pa', 'Dha', 'Ni♭'],
                'frequencies': [261.63, 277.18, 311.13, 349.23, 392.00, 440.00, 466.16],
                'characteristics': ['sad', 'longing', 'nostalgic', 'contemplative'],
                'time_preference': 'night',
                'tempo': 'slow'
            }
        }
    
    def _load_mood_keywords(self) -> Dict[str, List[str]]:
        """Load mood-indicating keywords in Tamil"""
        return {
            'peaceful': [
                'அமைதி', 'சாந்தம்', 'தியானம்', 'ஓம்', 'நிம்மதி', 'அமைதியான',
                'சமாதானம்', 'மௌனம்', 'தெய்வீகம்', 'இயற்கை', 'காலை', 'சூரியன்'
            ],
            'joyful': [
                'மகிழ்ச்சி', 'சந்தோஷம்', 'உற்சாகம்', 'கொண்டாட்டம்', 'மகிழ்வு',
                'உல்லாசம்', 'ஆனந்தம்', 'விழா', 'பண்டிகை', 'நடனம்', 'பாட்டு'
            ],
            'romantic': [
                'காதல்', 'அன்பு', 'இதயம்', 'பிரேமம்', 'நேசம்', 'பாசம்',
                'தோழி', 'தோழன்', 'கண்ணே', 'மனமே', 'இளமை', 'அழகு'
            ],
            'devotional': [
                'கடவுள்', 'பக்தி', 'ஆராதனை', 'பிரார்த்தனை', 'ஈஸ்வர்', 'சிவன்',
                'விஷ்ணு', 'தேவி', 'ஓம்', 'மந்திரம்', 'ஸ்லோகம்', 'கீர்த்தனை'
            ],
            'energetic': [
                'வீரம்', 'சக்தி', 'பலம்', 'தீரம்', 'உற்சாகம்', 'வேகம்',
                'போராட்டம்', 'வெற்றி', 'துணிவு', 'தைரியம்', 'விளையாட்டு'
            ],
            'melancholic': [
                'துக்கம்', 'வருத்தம்', 'கண்ணீர்', 'பிரிவு', 'தோல்வி', 'நினைவு',
                'தனிமை', 'விரக்தி', 'அழுகை', 'ஏக்கம்', 'மனவேதனை', 'பிரிவு'
            ]
        }
    
    def process_kavithai(self, kavithai: str) -> Dict:
        """Process Tamil kavithai for comprehensive music generation"""
        
        lines = kavithai.strip().split('\n')
        clean_lines = [line.strip() for line in lines if line.strip()]
        
        processed = {
            'original': kavithai,
            'lines': clean_lines,
            'total_lines': len(clean_lines),
            'syllable_count': self._count_syllables(kavithai),
            'word_count': self._count_words(kavithai),
            'mood_analysis': self._analyze_mood(kavithai),
            'phonetic': self._convert_to_phonetic(kavithai),
            'rhythm_pattern': self._analyze_rhythm(clean_lines),
            'poetic_structure': self._analyze_poetic_structure(clean_lines),
            'musical_elements': self._extract_musical_elements(kavithai),
            'emotional_intensity': self._calculate_emotional_intensity(kavithai),
            'recommended_raga': self._recommend_raga(kavithai),
            'tempo_suggestion': self._suggest_tempo(kavithai),
            'instrument_suggestions': self._suggest_instruments(kavithai)
        }
        
        return processed
    
    def _count_syllables(self, text: str) -> int:
        """Count syllables in Tamil text"""
        # Remove punctuation and spaces
        clean_text = re.sub(r'[^\u0B80-\u0BFF]', '', text)
        # Each Tamil character roughly represents one syllable
        return len(clean_text)
    
    def _count_words(self, text: str) -> int:
        """Count words in Tamil text"""
        words = re.findall(r'[\u0B80-\u0BFF]+', text)
        return len(words)
    
    def _analyze_mood(self, text: str) -> Dict:
        """Analyze mood based on Tamil keywords"""
        mood_scores = {mood: 0 for mood in self.mood_keywords.keys()}
        
        for mood, keywords in self.mood_keywords.items():
            for keyword in keywords:
                mood_scores[mood] += text.count(keyword)
        
        # Normalize scores
        total_score = sum(mood_scores.values())
        if total_score > 0:
            mood_percentages = {mood: (score / total_score) * 100 
                             for mood, score in mood_scores.items()}
        else:
            mood_percentages = {mood: 0 for mood in mood_scores.keys()}
        
        # Determine primary mood
        primary_mood = max(mood_percentages.items(), key=lambda x: x[1])
        
        return {
            'scores': mood_scores,
            'percentages': mood_percentages,
            'primary_mood': primary_mood[0],
            'confidence': primary_mood[1]
        }
    
    def _convert_to_phonetic(self, text: str) -> str:
        """Convert Tamil text to phonetic representation"""
        phonetic = ""
        i = 0
        
        while i < len(text):
            # Try to match longer combinations first
            matched = False
            
            for length in [2, 1]:  # Check 2-character then 1-character combinations
                if i + length <= len(text):
                    substring = text[i:i+length]
                    if substring in self.tamil_phonemes:
                        phonetic += self.tamil_phonemes[substring] + " "
                        i += length
                        matched = True
                        break
            
            if not matched:
                # If character not found in mapping, keep as is
                phonetic += text[i]
                i += 1
        
        return phonetic.strip()
    
    def _analyze_rhythm(self, lines: List[str]) -> Dict:
        """Analyze rhythm pattern of the kavithai"""
        if not lines:
            return self._default_rhythm_pattern()
        
        line_lengths = [len(line.replace(' ', '')) for line in lines]
        syllable_counts = [self._count_syllables(line) for line in lines]
        
        # Analyze patterns
        rhythm_analysis = {
            'line_count': len(lines),
            'avg_line_length': np.mean(line_lengths) if line_lengths else 0,
            'line_lengths': line_lengths,
            'syllable_counts': syllable_counts,
            'avg_syllables_per_line': np.mean(syllable_counts) if syllable_counts else 0,
            'rhythm_type': self._determine_rhythm_type(line_lengths),
            'beat_pattern': self._create_beat_pattern(syllable_counts),
            'meter': self._determine_meter(syllable_counts),
            'consistency_score': self._calculate_rhythm_consistency(syllable_counts)
        }
        
        return rhythm_analysis
    
    def _analyze_poetic_structure(self, lines: List[str]) -> Dict:
        """Analyze the poetic structure"""
        if not lines:
            return {'type': 'free_verse', 'stanzas': 0, 'pattern': None}
        
        # Detect stanza breaks (empty lines or pattern changes)
        stanzas = []
        current_stanza = []
        
        for line in lines:
            if line.strip():
                current_stanza.append(line)
            else:
                if current_stanza:
                    stanzas.append(current_stanza)
                    current_stanza = []
        
        if current_stanza:
            stanzas.append(current_stanza)
        
        # Analyze rhyme scheme (simplified)
        rhyme_pattern = self._detect_rhyme_pattern(lines)
        
        return {
            'type': self._classify_poem_type(lines),
            'stanzas': len(stanzas),
            'stanza_lengths': [len(stanza) for stanza in stanzas],
            'rhyme_pattern': rhyme_pattern,
            'structure_complexity': self._calculate_structure_complexity(stanzas)
        }
    
    def _extract_musical_elements(self, text: str) -> Dict:
        """Extract musical elements from the text"""
        musical_words = {
            'instruments': ['வீணை', 'தபேலா', 'புல்லாங்குழல்', 'மிருதங்கம்', 'நாதஸ்வரம்'],
            'musical_terms': ['இசை', 'பாட்டு', 'ராகம்', 'தாளம்', 'ஸ்வரம்', 'கீர்த்தனை'],
            'nature_sounds': ['பறவை', 'காற்று', 'நீர்', 'அலை', 'இடி', 'மழை'],
            'time_references': ['காலை', 'மாலை', 'இரவு', 'நடுராத்திரி', 'சூரியன்', 'சந்திரன்']
        }
        
        found_elements = {}
        for category, words in musical_words.items():
            found_elements[category] = [word for word in words if word in text]
        
        return found_elements
    
    def _calculate_emotional_intensity(self, text: str) -> float:
        """Calculate emotional intensity on a scale of 0-1"""
        intensity_indicators = {
            'high': ['!', '!!', '!!!', 'மிக', 'அதிக', 'பெரிய', 'உயர்ந்த'],
            'medium': ['?', 'என்ன', 'எப்படி', 'எதுவும்'],
            'low': ['.', ',', 'மெல்ல', 'அமைதியாக', 'மெதுவாக']
        }
        
        total_indicators = 0
        intensity_score = 0
        
        for level, indicators in intensity_indicators.items():
            count = sum(text.count(indicator) for indicator in indicators)
            total_indicators += count
            
            if level == 'high':
                intensity_score += count * 1.0
            elif level == 'medium':
                intensity_score += count * 0.6
            else:
                intensity_score += count * 0.3
        
        if total_indicators == 0:
            return 0.5  # Default medium intensity
        
        return min(intensity_score / total_indicators, 1.0)
    
    def _recommend_raga(self, text: str) -> str:
        """Recommend most suitable raga based on content analysis"""
        mood_analysis = self._analyze_mood(text)
        primary_mood = mood_analysis['primary_mood']
        
        # If no clear mood detected, analyze time references
        if mood_analysis['confidence'] < 20:
            time_words = {
                'morning': ['காலை', 'விடியல்', 'சூரியன்', 'உதயம்'],
                'afternoon': ['மதியம்', 'நண்பகல்', 'வெயில்'],
                'evening': ['மாலை', 'சாயங்காலம்', 'அஸ்தமனம்'],
                'night': ['இரவு', 'நிசி', 'சந்திரன்', 'நட்சத்திரம்']
            }
            
            for time_period, words in time_words.items():
                if any(word in text for word in words):
                    time_raga_map = {
                        'morning': 'peaceful',
                        'afternoon': 'joyful',
                        'evening': 'romantic',
                        'night': 'melancholic'
                    }
                    primary_mood = time_raga_map.get(time_period, 'peaceful')
                    break
        
        return primary_mood
    
    def _suggest_tempo(self, text: str) -> Dict:
        """Suggest tempo based on content analysis"""
        fast_indicators = ['ஓடு', 'வேகம்', 'விரைவு', 'துரிதம்', 'உற்சாகம்']
        slow_indicators = ['மெதுவாக', 'அமைதியாக', 'மெல்ல', 'நிதானம்']
        
        fast_count = sum(text.count(word) for word in fast_indicators)
        slow_count = sum(text.count(word) for word in slow_indicators)
        
        if fast_count > slow_count:
            return {'tempo': 'fast', 'bpm_range': '120-140', 'reasoning': 'energetic_content'}
        elif slow_count > fast_count:
            return {'tempo': 'slow', 'bpm_range': '60-80', 'reasoning': 'peaceful_content'}
        else:
            return {'tempo': 'medium', 'bpm_range': '80-120', 'reasoning': 'balanced_content'}
    
    def _suggest_instruments(self, text: str) -> List[str]:
        """Suggest instruments based on content analysis"""
        base_instruments = ['tabla', 'veena']  # Always include these
        
        # Add instruments based on content
        if any(word in text for word in ['புல்லாங்குழல்', 'காற்று', 'பறவை']):
            base_instruments.append('flute')
        
        if any(word in text for word in ['மிருதங்கம்', 'தாளம்', 'நடனம்']):
            base_instruments.append('mridangam')
        
        if any(word in text for word in ['கடவுள்', 'பக்தி', 'ஆராதனை']):
            base_instruments.extend(['nadhaswaram', 'thavil'])
        
        if any(word in text for word in ['இசை', 'ஸ்வரம்', 'ராகம்']):
            base_instruments.append('violin')
        
        # Always suggest keyboard for harmonic support
        if 'keyboard' not in base_instruments:
            base_instruments.append('keyboard')
        
        return list(set(base_instruments))  # Remove duplicates
    
    def create_music_prompt(self, processed_text: Dict, genre: str, mood: str, instruments: List[str]) -> str:
        """Create a detailed music prompt for generation"""
        
        # Start with base description
        prompt_parts = [f"Tamil {genre} music in {mood} mood"]
        
        # Add raga information
        if mood in self.raga_mappings:
            raga_info = self.raga_mappings[mood]
            prompt_parts.append(f"based on {raga_info['raga']} raga")
            prompt_parts.append(f"with {raga_info['tempo']} tempo")
        
        # Add instrument information
        instrument_descriptions = {
            'tabla': 'rhythmic tabla percussion',
            'veena': 'melodic veena strings', 
            'flute': 'flowing bamboo flute',
            'mridangam': 'traditional mridangam drums',
            'violin': 'expressive violin harmonies',
            'keyboard': 'harmonic keyboard foundation',
            'thavil': 'powerful thavil percussion',
            'nadhaswaram': 'majestic nadhaswaram wind'
        }
        
        featured_instruments = [instrument_descriptions.get(inst, inst) for inst in instruments]
        prompt_parts.append(f"featuring {', '.join(featured_instruments)}")
        
        # Add rhythm and structure information
        rhythm_info = processed_text.get('rhythm_pattern', {})
        if rhythm_info.get('meter'):
            prompt_parts.append(f"in {rhythm_info['meter']} meter")
        
        # Add emotional and musical characteristics
        if mood in self.raga_mappings:
            characteristics = self.raga_mappings[mood]['characteristics']
            prompt_parts.append(f"with {', '.join(characteristics[:2])} character")
        
        # Add cultural authenticity
        prompt_parts.append("authentic South Indian classical style")
        prompt_parts.append("professional recording quality")
        
        # Construct final prompt
        final_prompt = ", ".join(prompt_parts)
        
        # Limit prompt length for model compatibility
        if len(final_prompt) > 500:
            final_prompt = final_prompt[:497] + "..."
        
        return final_prompt
    
    def add_tamil_vocals(self, audio_path: str, kavithai: str, voice_type: str) -> str:
        """Add Tamil vocals to the generated music"""
        try:
            # Load the instrumental audio
            audio, sr = librosa.load(audio_path, sr=44100)
            
            # Process kavithai for vocal generation
            processed_kavithai = self.process_kavithai(kavithai)
            
            # Generate vocal melody
            vocal_melody = self._generate_vocal_melody(
                processed_kavithai, len(audio), sr, voice_type
            )
            
            # Mix vocals with instrumental
            mixed_audio = self._mix_vocals_with_instrumental(audio, vocal_melody)
            
            # Save mixed audio
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            sf.write(temp_file.name, mixed_audio, sr)
            
            self.logger.info(f"Vocals added successfully: {temp_file.name}")
            return temp_file.name
            
        except Exception as e:
            self.logger.error(f"Error adding vocals: {e}")
            return audio_path  # Return original if vocal addition fails
    
    def _generate_vocal_melody(self, processed_kavithai: Dict, audio_length: int, 
                              sr: int, voice_type: str) -> np.ndarray:
        """Generate a vocal melody based on processed kavithai"""
        
        # Create time array
        t = np.linspace(0, audio_length / sr, audio_length)
        
        # Get syllable information
        syllable_count = processed_kavithai.get('syllable_count', 20)
        lines = processed_kavithai.get('lines', [])
        
        # Determine vocal characteristics based on voice type
        voice_params = self._get_voice_parameters(voice_type)
        
        # Generate melody based on recommended raga
        recommended_raga = processed_kavithai.get('recommended_raga', 'peaceful')
        raga_info = self.raga_mappings.get(recommended_raga, self.raga_mappings['peaceful'])
        
        # Create vocal melody
        vocal_melody = self._synthesize_vocal_line(
            t, lines, raga_info, voice_params, sr
        )
        
        return vocal_melody
    
    def _get_voice_parameters(self, voice_type: str) -> Dict:
        """Get voice synthesis parameters"""
        params = {
            'male': {
                'base_freq': 130,
                'freq_range': (100, 200),
                'vibrato': 0.02,
                'formants': [500, 1500, 2500]
            },
            'female': {
                'base_freq': 220,
                'freq_range': (180, 300),
                'vibrato': 0.03,
                'formants': [600, 1800, 3000]
            },
            'child': {
                'base_freq': 300,
                'freq_range': (250, 400),
                'vibrato': 0.01,
                'formants': [700, 2000, 3500]
            },
            'chorus': {
                'base_freq': 175,
                'freq_range': (120, 280),
                'vibrato': 0.025,
                'formants': [550, 1650, 2750]
            }
        }
        
        return params.get(voice_type, params['male'])
    
    def _synthesize_vocal_line(self, t: np.ndarray, lines: List[str], 
                              raga_info: Dict, voice_params: Dict, sr: int) -> np.ndarray:
        """Synthesize vocal line with Tamil characteristics"""
        
        vocal_audio = np.zeros_like(t)
        
        if not lines:
            return vocal_audio
        
        # Calculate timing for each line
        total_duration = len(t) / sr
        time_per_line = total_duration / len(lines)
        
        frequencies = raga_info['frequencies']
        base_freq = voice_params['base_freq']
        
        for line_idx, line in enumerate(lines):
            start_time = line_idx * time_per_line
            end_time = (line_idx + 1) * time_per_line
            
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            
            if end_sample > len(vocal_audio):
                end_sample = len(vocal_audio)
            
            line_length = end_sample - start_sample
            if line_length <= 0:
                continue
            
            # Generate melody for this line
            line_t = np.linspace(0, time_per_line, line_length)
            line_audio = self._generate_line_melody(
                line_t, line, frequencies, voice_params
            )
            
            vocal_audio[start_sample:end_sample] = line_audio
        
        return vocal_audio * 0.3  # Keep vocals at moderate level
    
    def _generate_line_melody(self, t: np.ndarray, line: str, 
                             frequencies: List[float], voice_params: Dict) -> np.ndarray:
        """Generate melody for a single line"""
        
        if not line.strip():
            return np.zeros_like(t)
        
        # Simple syllable-based melody generation
        syllables = self._count_syllables(line)
        if syllables == 0:
            syllables = len(line.split())
        
        note_duration = len(t) / max(syllables, 1)
        melody = np.zeros_like(t)
        
        for i in range(syllables):
            start_idx = int(i * note_duration)
            end_idx = int((i + 1) * note_duration)
            
            if end_idx > len(melody):
                end_idx = len(melody)
            
            if start_idx >= end_idx:
                continue
            
            # Choose frequency from raga
            freq_idx = i % len(frequencies)
            base_freq = frequencies[freq_idx]
            
            # Adjust to voice range
            while base_freq < voice_params['freq_range'][0]:
                base_freq *= 2
            while base_freq > voice_params['freq_range'][1]:
                base_freq /= 2
            
            # Generate note
            note_t = t[start_idx:end_idx] - t[start_idx]
            note = np.sin(2 * np.pi * base_freq * note_t)
            
            # Add vibrato
            vibrato = 1 + voice_params['vibrato'] * np.sin(2 * np.pi * 5 * note_t)
            note *= vibrato
            
            # Add formants for voice-like quality
            for formant in voice_params['formants']:
                formant_component = 0.1 * np.sin(2 * np.pi * formant * note_t)
                note += formant_component
            
            # Apply envelope
            envelope = np.exp(-note_t * 2) * (1 - np.exp(-note_t * 10))
            note *= envelope
            
            melody[start_idx:end_idx] = note
        
        return melody
    
    def _mix_vocals_with_instrumental(self, instrumental: np.ndarray, 
                                    vocals: np.ndarray) -> np.ndarray:
        """Mix vocal and instrumental tracks"""
        
        # Ensure same length
        min_length = min(len(instrumental), len(vocals))
        instrumental = instrumental[:min_length]
        vocals = vocals[:min_length]
        
        # Mix with appropriate levels
        mixed = instrumental * 0.7 + vocals * 0.3
        
        # Normalize to prevent clipping
        max_amplitude = np.max(np.abs(mixed))
        if max_amplitude > 0.95:
            mixed = mixed * 0.95 / max_amplitude
        
        return mixed
    
    # Helper methods for rhythm and structure analysis
    def _default_rhythm_pattern(self):
        return {
            'line_count': 0,
            'avg_line_length': 0,
            'rhythm_type': 'free_verse',
            'beat_pattern': [4, 4, 4, 4],
            'meter': '4/4'
        }
    
    def _determine_rhythm_type(self, line_lengths: List[int]) -> str:
        if not line_lengths:
            return "free_verse"
        
        if len(set(line_lengths)) == 1:
            return "uniform"
        elif len(line_lengths) >= 4 and line_lengths[0] == line_lengths[2]:
            return "alternate"
        else:
            return "free_verse"
    
    def _create_beat_pattern(self, syllable_counts: List[int]) -> List[int]:
        if not syllable_counts:
            return [4, 4, 4, 4]
        
        beat_pattern = []
        for count in syllable_counts:
            if count <= 8:
                beat_pattern.append(3)
            elif count <= 16:
                beat_pattern.append(4)
            else:
                beat_pattern.append(6)
        
        return beat_pattern
    
    def _determine_meter(self, syllable_counts: List[int]) -> str:
        if not syllable_counts:
            return "4/4"
        
        avg_syllables = np.mean(syllable_counts)
        
        if avg_syllables <= 6:
            return "3/4"
        elif avg_syllables <= 12:
            return "4/4"
        else:
            return "6/8"
    
    def _calculate_rhythm_consistency(self, syllable_counts: List[int]) -> float:
        if len(syllable_counts) < 2:
            return 1.0
        
        variance = np.var(syllable_counts)
        mean_count = np.mean(syllable_counts)
        
        if mean_count == 0:
            return 0.0
        
        consistency = 1.0 / (1.0 + variance / mean_count)
        return consistency
    
    def _classify_poem_type(self, lines: List[str]) -> str:
        """Classify the type of Tamil poem"""
        
        if len(lines) <= 4:
            return "quatrain"
        elif len(lines) <= 8:
            return "octave"
        elif any("கடவுள்" in line or "பக்தி" in line for line in lines):
            return "devotional"
        elif any("காதல்" in line or "அன்பு" in line for line in lines):
            return "love_poem"
        else:
            return "free_verse"
    
    def _detect_rhyme_pattern(self, lines: List[str]) -> str:
        """Detect basic rhyme pattern (simplified)"""
        if len(lines) < 2:
            return "none"
        
        # Simple ending sound detection
        endings = [line[-2:] if len(line) >= 2 else line for line in lines]
        
        if len(set(endings)) == 1:
            return "monorhyme"
        elif len(lines) == 4 and endings[0] == endings[2] and endings[1] == endings[3]:
            return "ABAB"
        elif len(lines) >= 2 and endings[0] == endings[1]:
            return "couplet"
        else:
            return "free"
    
    def _calculate_structure_complexity(self, stanzas: List[List[str]]) -> float:
        """Calculate structural complexity score"""
        if not stanzas:
            return 0.0
        
        # Consider number of stanzas, variation in stanza length, etc.
        stanza_lengths = [len(stanza) for stanza in stanzas]
        length_variance = np.var(stanza_lengths) if len(stanza_lengths) > 1 else 0
        
        complexity = (len(stanzas) * 0.3 + length_variance * 0.7) / 10
        return min(complexity, 1.0)

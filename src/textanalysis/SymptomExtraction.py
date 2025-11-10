"""
LLM-Based Skin Symptom Extractor
Uses AI models for intelligent, context-aware extraction
"""

import json
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

# NOTE: For local Ollama, you'll need: pip install ollama
# Make sure Ollama is running locally: ollama serve

@dataclass
class SymptomExtraction:
    """Structured output from LLM extraction"""
    
    # Visual characteristics
    colors: List[str]  # red, pink, purple, etc.
    textures: List[str]  # bumpy, rough, scaly, smooth
    patterns: List[str]  # spotty, patchy, widespread
    size_descriptors: List[str]  # small, large, spreading
    
    # Physical sensations
    sensations: List[str]  # itchy, burning, painful
    severity: str  # mild, moderate, severe
    
    # Location
    body_locations: List[str]  # face, arms, legs, etc.
    
    # Temporal information
    duration: Optional[str]  # "3 days", "2 weeks", "chronic"
    progression: Optional[str]  # "getting worse", "improving", "stable"
    
    # Context
    triggers: List[str]  # sun exposure, food, stress, etc.
    aggravating_factors: List[str]  # heat, scratching, etc.
    relieving_factors: List[str]  # cold compress, cream, etc.
    
    # Additional observations
    associated_symptoms: List[str]  # fever, fatigue, etc.
    previous_treatments: List[str]  # mentioned creams, medications
    
    # Meta
    confidence: str  # high, medium, low
    key_concerns: List[str]  # What patient is most worried about
    original_text_snippets: List[str]  # Relevant quotes from conversation


class LLMSymptomExtractor:
    """
    Uses LLMs to extract skin symptoms from patient conversations.
    Supports multiple LLM providers.
    """
    
    def __init__(self, model_name: str = "gpt-oss:20b", base_url: str = "http://localhost:11434"):
        """
        Initialize with local Ollama model.
        
        Args:
            model_name: Name of the Ollama model (e.g., "gpt-oss:20b", "llama3", "mistral")
            base_url: Base URL for Ollama API (default: http://localhost:11434)
        """
        try:
            import ollama
        except ImportError:
            raise ImportError(
                "Ollama library not found. Install it with: pip install ollama\n"
                "Also make sure Ollama is running: ollama serve"
            )
        
        # Ollama client initialization
        if base_url != "http://localhost:11434":
            self.client = ollama.Client(host=base_url)
        else:
            self.client = ollama.Client()
        
        self.model = model_name
        self.provider = "ollama"
        
        # Test connection
        try:
            # Verify model is available
            models_list = self.client.list()
            model_names = [m['name'] for m in models_list.get('models', [])]
            if model_name not in model_names:
                print(f"⚠️  Warning: Model '{model_name}' not found in Ollama.")
                print(f"Available models: {', '.join(model_names) if model_names else 'None'}")
                print(f"Using '{model_name}' anyway - will attempt to use it.")
        except Exception as e:
            print(f"⚠️  Warning: Could not verify Ollama connection: {e}")
            print("Make sure Ollama is running: ollama serve")
    
    def create_extraction_prompt(self, conversation: str) -> str:
        """Create the prompt for LLM extraction"""
        
        prompt = f"""You are a medical AI assistant specialized in dermatology. Extract skin-related symptoms and information from the patient conversation below.

PATIENT CONVERSATION:
{conversation}

Extract the following information in JSON format:

{{
  "colors": ["list of color descriptors mentioned"],
  "textures": ["rough, smooth, bumpy, scaly, etc."],
  "patterns": ["spotty, patchy, widespread, etc."],
  "size_descriptors": ["small, large, spreading, etc."],
  "sensations": ["itchy, burning, painful, etc."],
  "severity": "mild/moderate/severe based on patient's description",
  "body_locations": ["specific body parts mentioned"],
  "duration": "how long they've had it",
  "progression": "getting worse/better/stable",
  "triggers": ["potential causes or triggers mentioned"],
  "aggravating_factors": ["what makes it worse"],
  "relieving_factors": ["what makes it better"],
  "associated_symptoms": ["fever, fatigue, other symptoms"],
  "previous_treatments": ["treatments or products they've tried"],
  "confidence": "high/medium/low - your confidence in the extraction",
  "key_concerns": ["main things the patient is worried about"],
  "original_text_snippets": ["relevant direct quotes from the conversation"]
}}

IMPORTANT INSTRUCTIONS:
- Only extract information that is EXPLICITLY mentioned or strongly implied
- If something is not mentioned, use an empty list [] or null
- Pay attention to negations (e.g., "not itchy" means DO NOT include itchy)
- Consider context and severity based on the patient's tone
- Capture the patient's own words in original_text_snippets
- Be precise with body locations
- If patient mentions timeframes, capture them exactly

Return ONLY the JSON object, no additional text."""

        return prompt
    
    def extract_with_ollama(self, conversation: str) -> SymptomExtraction:
        """Extract using local Ollama model"""
        prompt = self.create_extraction_prompt(conversation)
        
        try:
            # Use chat API for Ollama
            response = self.client.chat(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                options={
                    "temperature": 0,  # Deterministic for extraction
                    "num_predict": 2000,  # Max tokens
                }
            )
            
            # Extract response text
            response_text = response['message']['content']
            
            # Extract JSON from response (handles cases where LLM adds explanation)
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")
            
            json_str = response_text[json_start:json_end]
            data = json.loads(json_str)
            return SymptomExtraction(**data)
            
        except json.JSONDecodeError as e:
            print(f"⚠️  JSON parsing error: {e}")
            print(f"Response was: {response_text[:500]}...")
            raise
        except Exception as e:
            print(f"⚠️  Error calling Ollama: {e}")
            print("Make sure Ollama is running: ollama serve")
            print(f"Also verify model '{self.model}' is available: ollama pull {self.model}")
            raise
    
    def extract(self, conversation: str) -> SymptomExtraction:
        """
        Main extraction method using local Ollama model.
        
        Args:
            conversation: Patient conversation text
            
        Returns:
            SymptomExtraction object with all extracted information
        """
        if self.provider == "ollama":
            return self.extract_with_ollama(conversation)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def extract_batch(self, conversations: List[str]) -> List[SymptomExtraction]:
        """
        Extract from multiple conversations.
        
        Args:
            conversations: List of patient conversation texts
            
        Returns:
            List of SymptomExtraction objects
        """
        results = []
        for idx, conversation in enumerate(conversations):
            print(f"Processing conversation {idx + 1}/{len(conversations)}...")
            try:
                extraction = self.extract(conversation)
                results.append(extraction)
            except Exception as e:
                print(f"Error processing conversation {idx + 1}: {e}")
                # Add empty extraction on error
                results.append(self._empty_extraction())
        
        return results
    
    def _empty_extraction(self) -> SymptomExtraction:
        """Return empty extraction for error cases"""
        return SymptomExtraction(
            colors=[], textures=[], patterns=[], size_descriptors=[],
            sensations=[], severity="unknown", body_locations=[],
            duration=None, progression=None, triggers=[],
            aggravating_factors=[], relieving_factors=[],
            associated_symptoms=[], previous_treatments=[],
            confidence="low", key_concerns=[], original_text_snippets=[]
        )
    
    def aggregate_results(self, extractions: List[SymptomExtraction]) -> Dict:
        """
        Aggregate extraction results for analytics.
        
        Returns:
            Dictionary with frequency analysis and insights
        """
        from collections import Counter
        
        # Aggregate all symptoms
        all_colors = []
        all_textures = []
        all_sensations = []
        all_locations = []
        all_triggers = []
        severity_counts = []
        
        for ext in extractions:
            all_colors.extend(ext.colors)
            all_textures.extend(ext.textures)
            all_sensations.extend(ext.sensations)
            all_locations.extend(ext.body_locations)
            all_triggers.extend(ext.triggers)
            if ext.severity:
                severity_counts.append(ext.severity)
        
        return {
            "total_conversations": len(extractions),
            "most_common_colors": Counter(all_colors).most_common(10),
            "most_common_textures": Counter(all_textures).most_common(10),
            "most_common_sensations": Counter(all_sensations).most_common(10),
            "most_affected_locations": Counter(all_locations).most_common(10),
            "common_triggers": Counter(all_triggers).most_common(10),
            "severity_distribution": Counter(severity_counts),
            "high_confidence_extractions": sum(1 for e in extractions if e.confidence == "high"),
        }
    
    def generate_report(self, extractions: List[SymptomExtraction]) -> str:
        """Generate human-readable report"""
        agg = self.aggregate_results(extractions)
        
        report = "=" * 60 + "\n"
        report += "SKIN SYMPTOM ANALYSIS REPORT (LLM-Based)\n"
        report += "=" * 60 + "\n\n"
        report += f"Total Conversations Analyzed: {agg['total_conversations']}\n"
        report += f"High Confidence Extractions: {agg['high_confidence_extractions']}\n\n"
        
        report += "MOST COMMON SYMPTOMS:\n"
        report += "-" * 40 + "\n"
        for symptom, count in agg['most_common_sensations']:
            report += f"  • {symptom}: {count} times\n"
        
        report += "\nMOST COMMON VISUAL CHARACTERISTICS:\n"
        report += "-" * 40 + "\n"
        report += "Colors:\n"
        for color, count in agg['most_common_colors'][:5]:
            report += f"  • {color}: {count} times\n"
        report += "Textures:\n"
        for texture, count in agg['most_common_textures'][:5]:
            report += f"  • {texture}: {count} times\n"
        
        report += "\nMOST AFFECTED BODY LOCATIONS:\n"
        report += "-" * 40 + "\n"
        for location, count in agg['most_affected_locations']:
            report += f"  • {location}: {count} times\n"
        
        report += "\nSEVERITY DISTRIBUTION:\n"
        report += "-" * 40 + "\n"
        for severity, count in agg['severity_distribution'].items():
            report += f"  • {severity}: {count} cases\n"
        
        if agg['common_triggers']:
            report += "\nCOMMON TRIGGERS:\n"
            report += "-" * 40 + "\n"
            for trigger, count in agg['common_triggers']:
                report += f"  • {trigger}: {count} times\n"
        
        return report


# Example usage
if __name__ == "__main__":
    """
    SETUP INSTRUCTIONS:
    1. Install Ollama: https://ollama.ai
    2. Start Ollama server: ollama serve
    3. Pull your model: ollama pull gpt-oss:20b (or your preferred model)
    4. Install Python client: pip install ollama
    5. Run this script
    """
    
    # Sample conversations
    sample_conversations = [
        """I've had this rash on my arms for about 3 days now. It's super itchy, 
        especially at night. The spots are red with tiny bumps and they seem to be spreading. 
        I think it might have started after I went hiking and touched some plants. 
        I've tried hydrocortisone cream but it's not really helping.""",
        
        """My face has been breaking out really badly for the past 2 weeks. 
        I have these painful red bumps on my cheeks and forehead. Some of them have 
        white heads and they're really tender to touch. It's getting worse and I'm 
        really stressed about it. I've never had acne this bad before.""",
        
        """There's this weird patch on my leg that appeared last week. It's circular, 
        about the size of a quarter, and it's darker than my normal skin color. 
        It's not itchy or painful, just looks different. I'm worried because it seems 
        to be getting slightly bigger.""",
    ]
    
    # Initialize extractor with local Ollama
    # Default model is "gpt-oss:20b", you can change it:
    # extractor = LLMSymptomExtractor(model_name="llama3")  # or any other model
    extractor = LLMSymptomExtractor()  # Uses default "gpt-oss:20b"
    # Extract from single conversation (demo)
    print("Extracting from first conversation...\n")
    #result = extractor.extract(sample_conversations[0])
    #print("Generating a report")

    #print(json.dumps(asdict(result), indent=2))
    
    # For batch processing:
    results = extractor.extract_batch(sample_conversations)
    print(extractor.generate_report(results))
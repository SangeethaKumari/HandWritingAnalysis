import requests
import json

OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "alibayram/medgemma:latest"

def test_simple_prompt():
    """Test with a simple prompt"""
    print("="*70)
    print("TEST 1: Simple prompt")
    print("="*70)
    
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": MODEL_NAME,
        "prompt": "Say 'Hello, I am MedGemma!'",
        "stream": False
    }
    
    response = requests.post(url, json=payload)
    result = response.json()
    print("Response:", result.get("response", "No response"))
    print()

def test_json_format():
    """Test with JSON format specification"""
    print("="*70)
    print("TEST 2: JSON format")
    print("="*70)
    
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": MODEL_NAME,
        "prompt": 'Return only valid JSON: {"status": "ok", "message": "test"}',
        "stream": False,
        "format": "json"
    }
    
    response = requests.post(url, json=payload)
    result = response.json()
    raw_response = result.get("response", "")
    print("Raw response:", raw_response)
    
    try:
        parsed = json.loads(raw_response)
        print("✓ Successfully parsed as JSON:", parsed)
    except:
        print("✗ Could not parse as JSON")
    print()

def test_medical_prompt():
    """Test with a medical analysis prompt"""
    print("="*70)
    print("TEST 3: Medical analysis")
    print("="*70)
    
    prompt = """Analyze this symptom and respond in JSON format:
Patient says: "I have red itchy patches on my arm for 2 days"

Respond with this structure:
{
  "condition": "likely diagnosis",
  "severity": "emergency/moderate/minor",
  "recommended_action": "what to do"
}"""
    
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "format": "json"
    }
    
    response = requests.post(url, json=payload, timeout=120)
    result = response.json()
    raw_response = result.get("response", "")
    
    print("Raw response:")
    print(raw_response)
    print()
    
    # Try to extract and parse JSON
    json_text = raw_response
    if "```json" in json_text:
        start = json_text.find("```json") + 7
        end = json_text.find("```", start)
        if end != -1:
            json_text = json_text[start:end].strip()
    elif "```" in json_text:
        start = json_text.find("```") + 3
        end = json_text.find("```", start)
        if end != -1:
            json_text = json_text[start:end].strip()
    
    try:
        parsed = json.loads(json_text)
        print("✓ Successfully parsed:")
        print(json.dumps(parsed, indent=2))
    except Exception as e:
        print(f"✗ Could not parse: {e}")

if __name__ == "__main__":
    print("\n🔬 Testing MedGemma Setup\n")
    
    try:
        test_simple_prompt()
        test_json_format()
        test_medical_prompt()
        print("="*70)
        print("✓ All tests completed!")
        print("="*70)
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
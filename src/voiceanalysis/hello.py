#!/usr/bin/env python3
import requests
import json
import os
from pathlib import Path
from typing import Iterable, List

OLLAMA_URL = "http://localhost:11434"

def extract_text(text_field):
    """Extract text from various formats in Telegram JSON"""
    if isinstance(text_field, str):
        return text_field
    elif isinstance(text_field, list):
        result = ""
        for item in text_field:
            if isinstance(item, dict):
                result += item.get('text', '')
            elif isinstance(item, str):
                result += item
        return result
    return ""

def parse_conversations_json(conversation_data, person_a, person_b):
    """Parse Telegram JSON format conversation"""
    pairs = []
    
    try:
        data = json.loads(conversation_data)
        
        # Get messages array
        messages = data.get('messages', [])
        
        if not messages:
            return pairs
        
        # Extract pairs
        for i in range(len(messages) - 1):
            curr = messages[i]
            next_msg = messages[i + 1]
            
            # Get sender
            curr_sender = curr.get('from', '')
            next_sender = next_msg.get('from', '')
            
            # Get text
            curr_text = extract_text(curr.get('text', ''))
            next_text = extract_text(next_msg.get('text', ''))
            
            # Skip if no text or empty
            if not curr_text or not next_text:
                continue
            
            # Skip links and special content
            if curr_text.startswith('http') or next_text.startswith('http'):
                continue
            
            # Match person names
            if person_a.lower() in curr_sender.lower() and person_b.lower() in next_sender.lower():
                pairs.append({'personA': curr_text, 'personB': next_text})
        
        return pairs
    
    except json.JSONDecodeError:
        return []

def parse_conversations(conversation_text, person_a, person_b):
    """Parse conversation file - supports both TXT and JSON"""
    # Try JSON first
    json_pairs = parse_conversations_json(conversation_text, person_a, person_b)
    if json_pairs:
        return json_pairs
    
    # Fall back to text format
    lines = [line.strip() for line in conversation_text.split('\n') if line.strip()]
    pairs = []
    
    for i in range(len(lines) - 1):
        line = lines[i]
        next_line = lines[i + 1]
        
        if person_a in line and person_b in next_line:
            a_message = line.replace(person_a, '').replace(':', '').strip()
            b_message = next_line.replace(person_b, '').replace(':', '').strip()
            
            if a_message and b_message:
                pairs.append({'personA': a_message, 'personB': b_message})
    
    return pairs


def load_conversation_texts(source: str) -> List[str]:
    """Load one or more conversation files.

    `source` can be:
      * path to a single file
      * directory containing .json or .txt conversation files
      * comma-separated list of file paths
    """

    def _read_file(file_path: Path) -> str:
        try:
            return file_path.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"⚠️  Skipping {file_path}: {exc}")
            return ""

    path = Path(source).expanduser()
    texts: List[str] = []

    if path.is_dir():
        for child in sorted(path.iterdir()):
            if child.suffix.lower() in {".json", ".txt"} and child.is_file():
                texts.append(_read_file(child))
    elif path.is_file():
        texts.append(_read_file(path))
    else:
        # treat as comma-separated list
        for part in source.split(","):
            candidate = Path(part.strip()).expanduser()
            if candidate.is_file():
                texts.append(_read_file(candidate))
            else:
                print(f"⚠️  Path not found: {candidate}")

    return [text for text in texts if text]

def generate_response(user_message, training_pairs, person_a, person_b, model_name):
    """Generate response using Ollama"""
    
    # Define tone description and style anchors
    tone_description = (
        "Generally caring and soft-spoken, straightforward, and calm. "
        "Doesn’t diverge much from the topic. "
        "When the other person is feeling down or stuck, responds firmly, boldly, and with strong guidance "
        "to help them move forward."
    )

    style_anchors = (
        "Uses short sentences, empathetic phrasing, occasionally adds gentle encouragement. "
        "Rarely jokes, keeps it straightforward."
    )

    emotional_guidelines = (
        "Adjust response strength based on the other person’s mood: soft and caring normally, "
        "firm and guiding when the other person is down or discouraged."
    )

    personality_tags = "caring, straightforward, calm, firm when necessary, guiding, solution-focused"

    # Start building the training context
    training_context = f"""
    You are a language model that mimics {person_b}'s speaking style, reasoning, and preferences 
    based on past conversations.

    Guidelines for responses:
    - Tone: {tone_description}
    - Style: {style_anchors}
    - Emotional adaptation: {emotional_guidelines}
    - Personality: {personality_tags}
    - Keep responses natural, coherent, contextually appropriate, and concise.
    - Match {person_b}'s vocabulary, phrasing, and reasoning style.

    Past conversations:
    """

    # Add up to first 100 conversation pairs
    for pair in training_pairs[:100]:
        training_context += f"{person_a}: {pair['personA']}\n{person_b}: {pair['personB']}\n"

    # Final prompt with user message
    prompt = f"""{training_context}

    Now, {person_a} says: "{user_message}"
    Respond exactly as {person_b} would, matching tone, phrasing, and reasoning. 
    Adjust response based on the emotional context if relevant. 
    Keep it brief, natural, and contextually appropriate. Only provide the reply message, nothing else.
    """



    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.7
            },
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            return data.get('response', '').strip()
        else:
            return f"Error: {response.status_code}"
    
    except requests.exceptions.ConnectionError:
        return "Error: Cannot connect to Ollama. Make sure Ollama is running on http://localhost:11434"
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    print("\n" + "="*60)
    print("TELEGRAM CONVERSATION LLM CHAT")
    print("="*60 + "\n")
    
    default_path = "/home/sangeethagsk/agent_bootcamp/google_adk/voice_chatbot/conversations"
    source_path = default_path

    conversation_texts = load_conversation_texts(source_path)
    if not conversation_texts:
        print("No conversation files could be loaded. Check your path(s).")
        return

    person_a = input("Enter Person A name (who will message): ").strip()
    person_b = input("Enter Person B name (to mimic): ").strip()
    
    model_name = "llama3.1"
    
    training_pairs = []
    for conversation_data in conversation_texts:
        training_pairs.extend(parse_conversations(conversation_data, person_a, person_b))
    
    if not training_pairs:
        print(f"\nNo conversation pairs found. Check your file format.")
        print(f"Expected format: '{person_a}: message' followed by '{person_b}: response'")
        return
    
    print(f"\n✓ Loaded {len(conversation_texts)} file(s)")
    print(f"✓ Extracted {len(training_pairs)} conversation pairs")
    print(f"✓ Ready to chat as {person_b}\n")
    print("-"*60)
    print(f"Chat started! Type messages as {person_a}")
    print(f"Type 'quit' or 'exit' to end\n")
    print("-"*60 + "\n")
    
    while True:
        user_input = input(f"{person_a}: ").strip()
        
        if user_input.lower() in ['quit', 'exit']:
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        print(f"\n{person_b} is typing...", end='', flush=True)
        response = generate_response(user_input, training_pairs, person_a, person_b, model_name)
        print("\r" + " "*50 + "\r", end='')
        print(f"{person_b}: {response}\n")

if __name__ == "__main__":
    main()

import json
import os

def load_dataset(file_path):
    """Load the JSON dataset from file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def get_verse_context(data, current_index):
    """Get the previous, current, and next verse for context"""
    prev_verse = None
    next_verse = None
    current_verse = data[current_index]
    
    # Get previous verse
    if current_index > 0:
        prev_verse = data[current_index - 1]
    
    # Get next verse
    if current_index < len(data) - 1:
        next_verse = data[current_index + 1]
    
    return prev_verse, current_verse, next_verse

def format_context_verse(verse, verse_type="verse"):
    """Format context verse with Arabic and translation1 only"""
    if not verse:
        return f"No {verse_type} available"
    
    # Get Arabic text
    arabic_text = verse.get('arabic', 'No Arabic text available')
    
    # Get translation1 only
    translation1 = verse.get('translation1', 'No translation available')
    
    return f"Arabic: {arabic_text}\nTranslation: {translation1}"

def create_tags_prompt(prev_verse, target_verse, next_verse):
    """Create the prompt for OpenAI API to generate tags"""

    # Format context verses with Arabic and translation1 only
    prev_text = format_context_verse(prev_verse, "previous verse") if prev_verse else "No previous verse available"
    next_text = format_context_verse(next_verse, "next verse") if next_verse else "No next verse available"
    
    # Format target verse with all details
    target_arabic = target_verse.get('arabic', 'No Arabic text available')
    target_translation1 = target_verse.get('translation1', 'No translation1 available')
    target_translation2 = target_verse.get('translation2', 'No translation2 available')
    target_translation3 = target_verse.get('translation3', 'No translation3 available')
    
    # Get existing topics and subtopics
    topics = target_verse.get('topic', [])
    topics_str = ", ".join(topics) if topics else "No topics provided"
    
    subtopics = target_verse.get('subtopic', [])
    subtopics_str = ", ".join(subtopics) if subtopics else "No subtopics provided"

    prompt = f"""You are an expert Islamic scholar specializing in Quranic interpretation and semantic analysis. Your task is to generate concise and meaningful tags for a Target Quranic verse, making them relevant to general user queries about Islamic beliefs, practices, history, and ethical guidance.

You will be provided:
- The verse immediately before and after (with Arabic and one translation for context)
- The target verse with Arabic text, three translations, and existing scholarly topics/subtopics

## Objective:
Generate 2 tags for the Target Verse that will help users discover the verse through natural-language religious queries, such as:
- "What does Islam say about zakat?"
- "How were the Israelites saved in Islam?"
- "Are there Quranic stories of origin of life?"
- "What does the Quran say about women rights?"

These tags will be used in a Quran search tool that indexes verses semantically.

## Tag Guidelines:
- Use phrases that match likely user queries (e.g., "Quran on tyranny", "story of Israelites")
- Do not repeat the verse text; instead, summarize key themes in searchable language
- Consider:
  * Historical events (e.g., Israelites' rescue)
  * Ethical and social issues (e.g., persecution, gendered violence)
  * Sectors of society (e.g., women, children, oppressed)
  * Broader Islamic teachings (e.g., divine justice, trials of the believers)
- Tags should be specific, concise, and searchable
- Prioritize relevance to Islamic themes commonly asked about

## Context Information:

Previous Verse:
{prev_text}

Next Verse:
{next_text}

Target Verse:
Arabic: {target_arabic}
Translation1: {target_translation1}
Translation2: {target_translation2}
Translation3: {target_translation3}
Topics: {topics_str}
Subtopics: {subtopics_str}

## Output Format:
Return only valid JSON in this exact format:
{{
  "tags": ["tag1", "tag2"]
}}"""

    return prompt

def create_batch_file(input_file, output_file, surah_filter=None):
    """Create the batch file for OpenAI API to generate tags"""
    
    # Load dataset
    data = load_dataset(input_file)
    
    # Filter verses if surah_filter is provided
    if surah_filter:
        filtered_indices = []
        for i, verse in enumerate(data):
            if verse.get('id', '').startswith(f'{surah_filter}|'):
                filtered_indices.append(i)
        print(f"Found {len(filtered_indices)} verses with ID starting with '{surah_filter}|'")
    else:
        filtered_indices = list(range(len(data)))
        print(f"Processing all {len(filtered_indices)} verses")
    
    # Create batch requests
    batch_requests = []
    
    for current_index in filtered_indices:
        target_verse = data[current_index]
        
        # Get verse context
        prev_verse, current_verse, next_verse = get_verse_context(data, current_index)
        
        # Create prompt
        prompt = create_tags_prompt(prev_verse, target_verse, next_verse)
        
        # Create batch request
        batch_request = {
            "custom_id": target_verse['id'],
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4.1-2025-04-14",  
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert Islamic scholar specializing in Quranic interpretation. Always respond with valid JSON only."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 150,  # Reduced since we only need 2 tags
                "temperature": 0.1  # Slightly higher for creative tag generation
            }
        }
        
        batch_requests.append(batch_request)
    
    # Write to JSONL file
    with open(output_file, 'w', encoding='utf-8') as f:
        for request in batch_requests:
            f.write(json.dumps(request, ensure_ascii=False) + '\n')
    
    print(f"Created batch file: {output_file}")
    print(f"Total requests: {len(batch_requests)}")
    
    return len(batch_requests)

def main():
    # Configuration
    input_file = "dataset.json"  # Change this to your input file path
    output_file = "tags_batch_requests/Al-Faatiha_openai_tags_batch_requests.jsonl"  # Output batch file
    surah_filter = 1  # Change this to filter specific surah (e.g., "2" for Surah 2), or None for all verses
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        print("Please make sure your JSON dataset file is in the current directory.")
        return
    
    try:
        # Create batch file
        num_requests = create_batch_file(input_file, output_file, surah_filter)
        
        print(f"\nBatch file created successfully!")
        print(f"File: {output_file}")
        print(f"Number of requests: {num_requests}")
        print(f"\nYou can now upload this file to OpenAI's batch API.")
        
        # Display sample request for verification
        if num_requests > 0:
            print(f"\nSample request structure:")
            with open(output_file, 'r', encoding='utf-8') as f:
                sample_request = json.loads(f.readline())
                print(f"Custom ID: {sample_request['custom_id']}")
                print(f"Model: {sample_request['body']['model']}")
                print(f"Max tokens: {sample_request['body']['max_tokens']}")
                print(f"Temperature: {sample_request['body']['temperature']}")
        
    except Exception as e:
        print(f"Error creating batch file: {str(e)}")

if __name__ == "__main__":
    main()
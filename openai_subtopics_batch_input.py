import json
import os

def load_dataset(file_path):
    """Load the JSON dataset from file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def create_prompt(target_verse):
    """Create the prompt for OpenAI API to clean subtopics"""

    # Get all available translations
    translations = []
    for i in range(1, 4):
        trans_key = f'translation{i}'
        if trans_key in target_verse and target_verse[trans_key]:
            translations.append(target_verse[trans_key])

    translations_text = "\n".join([f"Translation {i}: \"{trans}\"" for i, trans in enumerate(translations, start=1)])

    # Include Arabic text
    arabic_text = target_verse.get('arabic', 'No Arabic text provided')

    # Format existing subtopics
    subtopics = target_verse.get('subtopic', [])
    subtopics_str = ", ".join(subtopics) if subtopics else "No subtopics provided"

    prompt = f"""You are an expert Islamic scholar specializing in Quranic interpretation and semantic analysis. Your task is to refine and clean subtopics for a given Quranic verse, ensuring maximum relevance, accuracy, and precision.

## Core Objectives:
### 1. Relevance Analysis
- **Direct Connection**: Keep only subtopics that directly relate to the verse's explicit content
- **Contextual Relevance**: Include subtopics that connect to the verse's broader thematic context
- **Semantic Precision**: Ensure subtopics capture the specific meaning, not just general concepts

### 2. Quality Control
- **Theological Accuracy**: Verify all subtopics align with authentic Islamic teachings
- **Linguistic Precision**: Correct spelling, grammar, and formatting
- **Conceptual Clarity**: Ensure subtopics are clear and unambiguous

### 3. Deduplication Process
- **Exact Duplicates**: Remove identical entries
- **Semantic Duplicates**: Merge similar concepts (e.g., "Adoption" and "Adopted sons" → "Adoption")
- **Hierarchical Duplicates**: Remove overly specific terms when broader terms exist
- **Linguistic Variations**: Consolidate Arabic/English variations of same concept

## Filtering Criteria:
### KEEP subtopics that are:
- **Explicitly mentioned** in the verse text
- **Core theological concepts** directly addressed
- **Key legal/jurisprudential terms** specifically relevant
- **Precise Arabic terminology** when more accurate than English
- **Actionable concepts** that users would search for

### REMOVE subtopics that are:
- **Too generic** (e.g., "Islam", "Quran", "God" unless specifically relevant)
- **Tangentially related** (mentioned in broader chapter context but not this verse)
- **Redundant** (covered by other, better subtopics)
- **Misspelled or poorly formatted**
- **Overly specific** without clear connection to verse meaning
- **Grammatical fragments** (e.g., "speaks the truth" when "Truth" suffices)

## Refinement Process:
1. **Extract Core Themes**: Identify 3-5 main concepts from the verse
2. **Map Subtopics**: Match existing subtopics to these core themes
3. **Consolidate Synonyms**: Merge related terms into single, best representation
4. **Verify Theological Accuracy**: Cross-check against Islamic scholarship
5. **Optimize for Searchability**: Ensure terms users would actually search for
6. **Prioritize Precision**: Prefer specific, accurate terms over generic ones

## Input Format:
Target Verse:

Arabic: [Arabic text]
Translation 1: [English translation]
Translation 2: [English translation]
Translation 3: [English translation] 
Existing Subtopics: [List of current subtopics]

## Output Requirements:
Return **only** valid JSON in this exact format:
{{
  "cleaned_subtopics": [
    "subtopic1",
    "subtopic2",
    "subtopic3"
  ]
}}
Quality Metrics:
Relevance Score: Each subtopic should have direct textual or thematic connection
Uniqueness: No semantic overlap between final subtopics
Precision: Specific enough to be meaningful, general enough to be searchable
Accuracy: Theologically sound and linguistically correct
Example Refinement:
Before: ["inheritance", "succession", "Mirath", "adapted sons", "Sons", "Child"]
After: ["Inheritance", "succession", "adapted sons"]

Remember: The goal is to create a precise, non-redundant set of subtopics that capture the verse's essential meaning and themes while being optimized for user search behavior.

Target Verse:

Arabic: {arabic_text}
Translations: {translations_text}
Existing Subtopics: {subtopics_str}
"""

    return prompt

def create_batch_file(input_file, output_file, surah_filter=None):
    """Create the batch file for OpenAI API"""
    
    # Load dataset
    data = load_dataset(input_file)
    
    # Filter verses if surah_filter is provided
    filtered_data = []
    if surah_filter:
        for i, verse in enumerate(data):
            if verse.get('id', '').startswith(f'{surah_filter}|'):
                filtered_data.append((i, verse))
        print(f"Found {len(filtered_data)} verses with ID starting with '{surah_filter}|'")
    else:
        filtered_data = [(i, verse) for i, verse in enumerate(data)]
        print(f"Processing all {len(filtered_data)} verses")
    
    # Create batch requests
    batch_requests = []
    
    for original_index, target_verse in filtered_data:
        # Skip if no subtopics exist
        if 'subtopic' not in target_verse or not target_verse['subtopic']:
            continue
            
        # Create prompt
        prompt = create_prompt(target_verse)
        
        # Create batch request
        batch_request = {
            "custom_id": target_verse['id'],
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4.1-2025-04-14",  # You can change this to your preferred model
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert Islamic scholar. Always respond with valid JSON only."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 300,
                "temperature": 0.1  # Low temperature for consistent results
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
    output_file = "_openai_subtopic_batch_requests.jsonl"  # Output batch file
    surah_filter = "114"  # Change this to filter specific surah (e.g., "2" for Surah 2), or None for all verses
    
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
        
    except Exception as e:
        print(f"Error creating batch file: {str(e)}")

if __name__ == "__main__":
    main()
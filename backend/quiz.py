import json
import re
import ollama

def generate_quiz_data(transcript_path="data/transcripts/test_transcript.json"):
    """
    Generates structured quiz data from the transcript.
    Returns:
        A list of dictionaries, each containing:
        - question (str)
        - options (dict with keys A, B, C, D)
        - correct_answer (str: 'A', 'B', 'C', or 'D')
    """
    try:
        with open(transcript_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Handle if transcript is list of chunks or full text
            if isinstance(data, list):
                full_text = " ".join([chunk.get("text", "") for chunk in data])
            else:
                full_text = str(data)
    except Exception as e:
        print(f"Error loading transcript: {e}")
        return []

    # Truncate to avoid context limit issues if necessary
    display_text = full_text[:20000]

    prompt = f"""
    Based on the following transcript, generate exactly 5 multiple choice questions (MCQs) to test understanding.
    
    Transcript:
    {display_text}
    
    Output Format:
    For each question, use exactly this format:
    
    Question: <Question Text>
    A) <Option A>
    B) <Option B>
    C) <Option C>
    D) <Option D>
    Correct Answer: <Option Letter>
    
    Separate each question with a blank line. Do not output anything else.
    """

    try:
        response = ollama.chat(
            model="qwen2.5:7b",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.7} 
        )
        content = response["message"]["content"]
    except Exception as e:
        print(f"Error generating quiz: {e}")
        return []

    # Parse the response
    questions = []
    # Regex to capture the structure
    # Matches Question: ... A) ... B) ... C) ... D) ... Correct Answer: X
    block_pattern = r"Question:\s*(.+?)\s*A\)\s*(.+?)\s*B\)\s*(.+?)\s*C\)\s*(.+?)\s*D\)\s*(.+?)\s*Correct Answer:\s*([A-D])"
    
    matches = re.findall(block_pattern, content, re.DOTALL | re.IGNORECASE)

    for match in matches:
        q_text, a, b, c, d, ans = match
        questions.append({
            "question": q_text.strip(),
            "options": {
                "A": a.strip(),
                "B": b.strip(),
                "C": c.strip(),
                "D": d.strip()
            },
            "correct_answer": ans.strip().upper()
        })

    return questions

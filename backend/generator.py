import ollama


def generate_answer(query, context):

    prompt = f"""
You are a grounded retrieval assistant.

Use ONLY the provided context to answer the question.

Rules:
- Do NOT add outside knowledge.
- If the answer is not in the context, reply exactly:
  "Not found in transcript."
- Respond in full sentence format.
- If listing items, combine them into one clear sentence.

Context:
{context}

Question:
{query}

Answer in complete sentence:
"""

    response = ollama.chat(
        model="qwen2.5:7b",
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": 0,
            "top_p": 1
        }
    )

    return response["message"]["content"].strip()

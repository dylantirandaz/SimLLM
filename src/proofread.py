import openai
import time
from src.config import Config

openai.api_key = Config.OPENAI_API_KEY

def proofread_text(text: str, max_retries=3) -> str:
    """
    Proofread text w/ChatGPT
    """
    for _ in range(max_retries):
        try:
            response = openai.chat.completions.create(
                model=Config.OPENAI_MODEL,
                messages=[{"role": "user", "content": f"Proofread this text: {text}"}],
                temperature=0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error: {e}. Retrying...")
            time.sleep(5)
    return text  

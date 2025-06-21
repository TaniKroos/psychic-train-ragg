# cohere_client.py

import cohere
from typing import List
from dotenv import load_dotenv
import os
load_dotenv ()
COHERE_API_KEY = os.getenv("CO_API_KEY")

print("Initializing Cohere client...")
print("Loaded CO_API_KEY:", COHERE_API_KEY)


# Choose models (you can modify these)
EMBED_MODEL = "embed-v4.0"
GENERATE_MODEL = "command-r-plus"

class CohereClient:
    def __init__(self, api_key: str = COHERE_API_KEY):

        if not api_key:
            raise ValueError("Cohere API key is missing. Set CO_API_KEY in env.")
        self.client = cohere.Client(api_key)

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a list of text chunks.
        """
        response = self.client.embed(
            texts=texts,
            model=EMBED_MODEL,
            input_type="search_document"
        )
        return response.embeddings

    def generate_response(self, query: str, context: str) -> str:
        """
        Generate a response using the Cohere LLM.
        """
        prompt = f"""You are an assistant answering questions using the following context:\n\n{context}\n\nQuestion: {query}\nAnswer:"""
        response = self.client.generate(
            model=GENERATE_MODEL,
            prompt=prompt,
            max_tokens=300,
            temperature=0.3
        )
        return response.generations[0].text.strip()

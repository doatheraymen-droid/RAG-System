from abc import ABC, abstractmethod
import requests
import os

# Abstract Base Class (Interface) for Bonus 1
class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass
    
    def health_check(self):
        return True

# Ollama Provider (Local, Free)
class OllamaProvider(LLMProvider):
    def __init__(self):
        self.url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    
    def generate(self, prompt: str) -> str:
        try:
            response = requests.post(
                f"{self.url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=60
            )
            if response.status_code == 200:
                return response.json().get("response", "Error generating response")
            return f"Ollama error: {response.status_code}"
        except Exception as e:
            return f"Error connecting to Ollama: {e}"
    
    def health_check(self):
        try:
            response = requests.get(f"{self.url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

# OpenAI Provider (Paid, Better Quality)
class OpenAIProvider(LLMProvider):
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    
    def generate(self, prompt: str) -> str:
        if not self.api_key:
            return "OpenAI API key not set. Please add OPENAI_API_KEY to .env file."
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1
        }
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=60
            )
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            return f"OpenAI error: {response.status_code}"
        except Exception as e:
            return f"Error connecting to OpenAI: {e}"

# Hugging Face Provider (Free, Rate Limited)
class HuggingFaceProvider(LLMProvider):
    def __init__(self):
        self.api_key = os.getenv("HUGGINGFACE_API_KEY")
        self.model = os.getenv("HF_MODEL", "google/flan-t5-small")
    
    def generate(self, prompt: str) -> str:
        if not self.api_key:
            return "HuggingFace API key not set. Please add HUGGINGFACE_API_KEY to .env file."
        
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            response = requests.post(
                f"https://api-inference.huggingface.co/models/{self.model}",
                headers=headers,
                json={"inputs": prompt, "parameters": {"max_new_tokens": 200}},
                timeout=60
            )
            if response.status_code == 200:
                return response.json()[0]["generated_text"]
            return f"HuggingFace error: {response.status_code}"
        except Exception as e:
            return f"Error connecting to HuggingFace: {e}"

# Factory Class for Bonus 1
class LLMFactory:
    @staticmethod
    def get_provider(provider_name: str = None) -> LLMProvider:
        provider = provider_name or os.getenv("LLM_PROVIDER", "ollama")
        
        if provider == "ollama":
            return OllamaProvider()
        elif provider == "openai":
            return OpenAIProvider()
        elif provider == "huggingface":
            return HuggingFaceProvider()
        else:
            print(f"Unknown provider: {provider}. Using Ollama as default.")
            return OllamaProvider()
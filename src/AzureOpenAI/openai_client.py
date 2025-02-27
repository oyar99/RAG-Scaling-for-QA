from openai import AzureOpenAI
import os

from utils.singleton import Singleton

class OpenAIClient(metaclass=Singleton):
    """
    A class that provides a singleton instance of the Azure OpenAI client.
    """
    def __init__(self):
        self._initialize_client()

    def _initialize_client(self):
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION") or "2024-08-01-preview"
        
        self._client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version
        )

    def get_client(self):
        """Returns the Azure OpenAI client.

        Returns:
            AzureOpenAI: A singleton instance of the Azure OpenAI client.
        """
        return self._client
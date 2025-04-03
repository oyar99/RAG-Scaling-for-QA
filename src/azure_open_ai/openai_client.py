"""A module to create a singleton instance of the Azure OpenAI client."""
import os
from openai import AzureOpenAI

from utils.singleton import Singleton


class OpenAIClient(metaclass=Singleton):
    """
    A class that provides a singleton instance of the Azure OpenAI client.
    """

    def __init__(self):
        self._client = None
        self.initialize_client()

    def initialize_client(self):
        """
        Initializes the Azure OpenAI client if it is not already initialized.
        
        Raises:
            ValueError: if any of the required environment variables are not set
        """
        if self._client is None:
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            api_version = os.getenv(
                "AZURE_OPENAI_API_VERSION") or "2024-12-01-preview"

            self._client = AzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=api_key,
                api_version=api_version
            )

    def get_client(self):
        """
        Returns the Azure OpenAI client.

        Returns:
            client (AzureOpenAI): A singleton instance of the Azure OpenAI client.
        """
        return self._client

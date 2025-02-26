from openai import AzureOpenAI
import os

openAI_client = None

def init_openai_client():
    """
    Initializes the Azure OpenAI client.

    Returns:
        AzureOpenAI: An instance of the AzureOpenAI client.
    """
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION") or "2024-08-01-preview"
    
    global openAI_client
    
    openAI_client = AzureOpenAI(
    azure_endpoint=azure_endpoint,
    api_key=api_key,
    api_version=api_version
)
    
def get_openai_client():
    """
    Gets the Azure OpenAI client.

    Returns:
        AzureOpenAI: An instance of the AzureOpenAI client.
    """
    if not openAI_client:
        init_openai_client()
        
    return openAI_client
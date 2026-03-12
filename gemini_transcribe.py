import argparse
from google import genai
from google.genai import types


class GeminiModelConfig:
    """
    Configuration class for Gemini Pro 2.5 model with custom settings.
    """
    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.5-pro",
        temperature: float = 0.1,
        top_k: int = 1,
        top_p: float = 1.0,
        max_output_tokens: int = 8192,
        use_grounding: bool = True
    ):
        """
        Initialize Gemini model configuration.
        
        Args:
            api_key: Your Gemini API key
            model_name: Model name (default: gemini-2.5-pro)
            temperature: Controls randomness (0.0-2.0, lower is more deterministic)
            top_k: Limits token selection to top K tokens
            top_p: Nucleus sampling threshold
            max_output_tokens: Maximum tokens in response
            use_grounding: Enable Google Search grounding
        """
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.max_output_tokens = max_output_tokens
        self.use_grounding = use_grounding
        
        # Initialize the client
        self.client = genai.Client(api_key=self.api_key)
        
        # Set up generation config
        self.generation_config = types.GenerateContentConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_output_tokens=self.max_output_tokens,
        )
        
        # Add grounding if enabled
        if self.use_grounding:
            self.generation_config.tools = [types.Tool(google_search=types.GoogleSearch())]
        
        # Set safety settings to be permissive
        self.generation_config.safety_settings = [
            types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="BLOCK_NONE"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="BLOCK_NONE"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="BLOCK_NONE"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="BLOCK_NONE"
            ),
        ]
    
    def get_client(self):
        """Return the initialized client."""
        return self.client
    
    def get_config(self):
        """Return the generation config."""
        return self.generation_config


# Configuration constants (env for the key)
GEMINI_API_KEY = ""
TRANSCRIPTION_PROMPT = "You are an expert linguist and professional transcriber with deep knowledge of multilingual speech patterns, regional accents, and code-switching. You will listen carefully to the provided audio file and produce an accurate, verbatim transcription in the original languages used by the speakers."

def transcribe_with_gemini(audio_file_path, output_txt_path):
    """
    Transcribes the given audio file via Gemini 2.5 Pro using a template prompt, saves output to text file.
    
    Args:
        audio_file_path: Path to the audio file
        output_txt_path: Path to save the transcription
    """
    # Initialize Gemini model with custom configuration
    gemini_config = GeminiModelConfig(api_key=GEMINI_API_KEY)
    client = gemini_config.get_client()
    config = gemini_config.get_config()
    
    # Upload the audio file
    print(f"Uploading audio file: {audio_file_path}")
    audio_file = client.files.upload(file=audio_file_path)
    print(f"File uploaded successfully: {audio_file.name}")
    
    # Create the prompt
    full_prompt = f"{TRANSCRIPTION_PROMPT}\n\nPlease transcribe the following audio file accurately."
    
    # Generate transcription
    print("Generating transcription...")
    response = client.models.generate_content(
        model=gemini_config.model_name,
        contents=[full_prompt, audio_file],
        config=config
    )
    
    # Extract transcription
    transcription = response.text
    
    if not transcription:
        raise RuntimeError("No transcript generated from the model!")
    
    # Save to file
    with open(output_txt_path, "w", encoding="utf-8") as out_file:
        out_file.write(transcription)
    
    print(f"Transcription saved to {output_txt_path}")
    
    # Clean up uploaded file
    client.files.delete(name=audio_file.name)
    print("Temporary file cleaned up.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio file using Gemini 2.5 Pro and save to text file.")
    parser.add_argument("audio_file", type=str, help="Path to the audio file (e.g., WAV/MP3)")
    parser.add_argument("output_txt", type=str, help="Path for output text file")

    args = parser.parse_args()
    transcribe_with_gemini(args.audio_file, args.output_txt)

from openai import OpenAI

def tokenize_text(text):
    """
    Tokenizes the input text using LMStudio's local server.

    Args:
        text (str): The text to tokenize.

    Returns:
        list: A list of tokens.
    """
    # Create a client pointing to LMStudio's local server
    client = OpenAI(
        base_url="http://localhost:1234/v1",  # LMStudio default address
        api_key="not-needed"  # LMStudio does not require an API key, but it must be passed
    )

    # Send a request to tokenize the text
    response = client.chat.completions.create(
        model="local-model",  # This value is ignored; LMStudio uses the loaded model
        messages=[
            {"role": "system", "content": "Tokenize the following text."},
            {"role": "user", "content": text}
        ],
        temperature=0.0,  # Ensure deterministic tokenization
        max_tokens=256
    )

    # Extract tokens from the response
    tokens = response.choices[0].message.content.split()

    return tokens

if __name__ == "__main__":
    sample_text = "This is a sample text to tokenize."
    tokens = tokenize_text(sample_text)
    print("Tokens:", tokens)
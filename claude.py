from anthropic import Anthropic

# Initialize client (reads API key from env var)
client = Anthropic(api_key="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")

# Make a request
response = client.messages.create(
    model="claude-3-7-sonnet-20250219",  # or claude-3-opus / claude-3-haiku
    max_tokens=200,
    temperature=0.7,
    messages=[
        {"role": "user", "content": "Explain Monte Carlo Tree Search in simple terms."}
    ]
)

# Print assistant’s reply
print(response.content[0].text)

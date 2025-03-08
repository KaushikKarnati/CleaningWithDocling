import google.generativeai as genai

# Configure the API (Make sure to replace with your actual API key)
genai.configure(api_key="AIzaSyAmjkt50bwTSY0FsWWXN6mEAorhfxRDCPw")

# List available models
models = genai.list_models()

# Print available models
for model in models:
    print(f"Model Name: {model.name} | Supported Methods: {model.supported_generation_methods}")
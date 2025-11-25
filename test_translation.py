from deep_translator import GoogleTranslator
import time

try:
    print("Initializing translator...")
    translator = GoogleTranslator(source='en', target='vi')
    print("Translator initialized.")
    
    text = "Hello world"
    print(f"Translating: {text}")
    translated = translator.translate(text)
    print(f"Result: {translated}")
    
    text2 = "This is a test of the emergency broadcast system."
    print(f"Translating: {text2}")
    translated2 = translator.translate(text2)
    print(f"Result: {translated2}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

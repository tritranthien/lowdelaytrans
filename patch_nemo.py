import os

file_path = r"c:\laragon\bin\python\python-3.10\lib\site-packages\nemo\collections\common\tokenizers\__init__.py"

if os.path.exists(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Check if already patched
    if "class YouTokenToMeTokenizer:" in content and "raise ImportError" in content:
        print("Already patched.")
    else:
        # The line to replace
        target = "from nemo.collections.common.tokenizers.youtokentome_tokenizer import YouTokenToMeTokenizer"
        
        replacement = """try:
    from nemo.collections.common.tokenizers.youtokentome_tokenizer import YouTokenToMeTokenizer
except (ImportError, ModuleNotFoundError):
    class YouTokenToMeTokenizer:
        def __init__(self, *args, **kwargs):
            raise ImportError("YouTokenToMeTokenizer is not available because youtokentome could not be installed.")"""
            
        if target in content:
            new_content = content.replace(target, replacement)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            print("Successfully patched NeMo tokenizers.")
        else:
            print("Target string not found in file.")
            print("Content preview:", content[:500])
else:
    print(f"File not found: {file_path}")

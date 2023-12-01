def custom_vectorizer(text):
    for key, value in vector.items():
        # Split the key into variations
        variations = [v.strip() for v in key.split(',')]
        
        # Create a regular expression pattern with word boundaries for each variation
        pattern = '|'.join(r'\b' + re.escape(v) + r'\b' for v in variations)
        
        # Use re.sub with the pattern for replacement
        text = re.sub(pattern, value, text, flags=re.IGNORECASE)
        
    return text.lower()
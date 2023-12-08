import joblib
import numpy as np

# Load the spam detection model
loaded_model = joblib.load('spam_model.joblib')

# Example text to test
test_text = "Experience BER Month delight! Enjoy free Golden Eggs, a Php 7,777 deposit bonus, and a Php20 app download bonus at http://JackpotCity.vegas"


# Load the original vectorizer used during training
vectorizer = joblib.load('vectorizer.joblib')  # Replace 'vectorizer.joblib' with the actual filename you used to save the vectorizer during training

# Transform the test text using the original vectorizer
test_text_transformed = vectorizer.transform([test_text])

# Predict if the text is spam or not
prediction = loaded_model.predict(test_text_transformed)
# Get probability scores for each class
probabilities = loaded_model.predict_proba(test_text_transformed)

# Probability of being spam
spam_probability = probabilities[0, 1]

# Set a threshold from Precision-Recall Curve
threshold = 0.6568449587313073

# Check the prediction based on the threshold
print(test_text)
if spam_probability >= threshold:
    print("The text is likely spam.")
else:
    print("The text is likely not spam.")




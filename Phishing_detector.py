import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

#  Improved training data
data = {
    "text": [
        # Phishing emails (1)
        "Reset your password immediately",
        "Congratulations you won a prize click here",
        "Your account has been suspended login now",
        "Click here to verify your bank account",
        "You received a prize click the link",
        "Urgent action required update your payment info",
        "Security alert confirm your identity now",
        "Claim your free reward now",
        "Login now to avoid account suspension",

        # Safe emails (0)
        "Meeting scheduled for tomorrow",
        "Project report attached",
        "Let's have lunch today",
        "See you at the meeting",
        "Homework submission deadline",
        "Happy birthday have a great day",
        "Call me when you arrive",
        "Family dinner tonight",
        "Here is the document you asked for"
    ],
    "label": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
}

df = pd.DataFrame(data)

#  Convert text into numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["label"]

#  Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the AI model
model = MultinomialNB()
model.fit(X_train, y_train)

# Check accuracy
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("AI model is ready! Type email text to test.\n")

# Real-time email checking
while True:
    email = input("Enter email text (or type 'exit'): ")

    if email.lower() == "exit":
        print("Program stopped.")
        break

    email_vector = vectorizer.transform([email])
    prediction = model.predict(email_vector)[0]

    if prediction == 1:
        print("⚠️ This is likely a PHISHING email!\n")
    else:
        print("✅ This looks like a SAFE email.\n")

import cv2
import pytesseract
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
import matplotlib.pyplot as plt

# ========================= MODEL TRAINING =========================
data = pd.read_csv("spam.csv", encoding='latin-1')[['Category', 'Message']]
data.columns = ['Category', 'Text']
data['Label'] = data['Category'].apply(lambda x: 1 if x == 'spam' else 0)

X_train, X_test, y_train, y_test = train_test_split(
    data['Text'], data['Label'], test_size=0.2, random_state=0
)

tfidf = TfidfVectorizer(stop_words='english', lowercase=True)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)
y_pred = nb.predict(X_test_tfidf)

accuracy = round(accuracy_score(y_test, y_pred), 3)
precision = round(precision_score(y_test, y_pred), 3)
recall = round(recall_score(y_test, y_pred), 3)

print("Model Performance:")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")

# -------- Plot model performance graph --------
metrics = ['Accuracy', 'Precision', 'Recall']
values = [accuracy, precision, recall]

plt.figure(figsize=(7,5))
plt.bar(metrics, values, color=['skyblue', 'lightgreen', 'salmon'])
plt.title('MailGuard Model Performance Metrics', pad=15)
plt.tight_layout()
plt.show()

# ========================= FUNCTIONS =========================
def recognize_text(image_path):
    try:
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        config = '--psm 6'
        text = pytesseract.image_to_string(thresh, config=config)
        return text.strip() if text.strip() else "Could not recognize text"
    except Exception as e:
        return f"Error during OCR: {str(e)}"

def show_prediction_graph(result):
    plt.figure(figsize=(4,3))
    color = 'red' if result == "Spam" else 'green'
    plt.bar(["Email Result"], [1], color=color)
    plt.title(f"Prediction Result: {result}")
    plt.ylim(0, 1.5)
    plt.text(0, 1.05, result, ha='center', fontweight='bold')
    plt.tight_layout()
    plt.show()

def analyze_text_input():
    text = text_box.get("1.0", tk.END).strip()
    if not text:
        messagebox.showwarning("Empty Input", "Please enter some text for analysis.")
        return
    vec = tfidf.transform([text])
    pred = nb.predict(vec)[0]
    result = "Spam" if pred == 1 else "Safe"

    result_box.config(state='normal')
    result_box.delete("1.0", tk.END)

    if result == "Spam":
        result_box.insert(tk.END, f"Prediction: {result}", "spam")
    else:
        result_box.insert(tk.END, f"Prediction: {result}", "safe")

    result_box.tag_config("spam", foreground="red", font=("Segoe UI", 12, "bold"))
    result_box.tag_config("safe", foreground="green", font=("Segoe UI", 12, "bold"))
    result_box.config(state='disabled')

    show_prediction_graph(result)

def analyze_image_input():
    file_path = filedialog.askopenfilename(
        title="Select Email Screenshot",
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")]
    )
    if not file_path:
        return
    text = recognize_text(file_path)
    if "Could not" in text:
        messagebox.showinfo("OCR Result", "No readable text found in the image.")
        return
    vec = tfidf.transform([text])
    pred = nb.predict(vec)[0]
    result = "Spam" if pred == 1 else "Safe"

    result_box.config(state='normal')
    result_box.delete("1.0", tk.END)
    result_box.insert(tk.END, f"Extracted Text:\n{text[:300]}...\n\n", "normal")

    if result == "Spam":
        result_box.insert(tk.END, f"Prediction: {result}", "spam")
    else:
        result_box.insert(tk.END, f"Prediction: {result}", "safe")

    result_box.tag_config("spam", foreground="red", font=("Segoe UI", 12, "bold"))
    result_box.tag_config("safe", foreground="green", font=("Segoe UI", 12, "bold"))
    result_box.tag_config("normal", foreground="black", font=("Consolas", 10))
    result_box.config(state='disabled')

    show_prediction_graph(result)

# ========================= UI DESIGN =========================
root = tk.Tk()
root.title("MailGuard: Spam Detection System")
root.geometry("750x600")
root.configure(bg="#f9f9f9")

# --- Title ---
title = tk.Label(root, text="MailGuard: Spam Detection", font=("Segoe UI", 18, "bold"), bg="#f9f9f9", fg="black")
title.pack(pady=15)

# --- Button Frame ---
btn_frame = tk.Frame(root, bg="#f9f9f9")
btn_frame.pack(pady=10)

text_btn = tk.Button(
    btn_frame,
    text="Enter Text for Analysis",
    font=("Segoe UI", 11, "bold"),
    bg="black",
    fg="white",
    activebackground="#333333",
    activeforeground="white",
    relief="flat",
    width=22,
    command=lambda: text_box.focus()
)
text_btn.grid(row=0, column=0, padx=10)

image_btn = tk.Button(
    btn_frame,
    text="Upload SS for Analysis",
    font=("Segoe UI", 11, "bold"),
    bg="black",
    fg="white",
    activebackground="#333333",
    activeforeground="white",
    relief="flat",
    width=22,
    command=analyze_image_input
)
image_btn.grid(row=0, column=1, padx=10)

# --- Text Input Section ---
text_label = tk.Label(root, text="Enter Email Text:", font=("Segoe UI", 12, "bold"), bg="#f9f9f9", fg="black")
text_label.pack(anchor="w", padx=40, pady=(20, 5))

text_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=10, font=("Consolas", 10))
text_box.pack(padx=40, pady=5, ipady=5)
text_box.config(relief="solid", bd=1)

# --- Analyze Button ---
analyze_btn = tk.Button(
    root,
    text="Analyze Text",
    font=("Segoe UI", 11, "bold"),
    bg="green",
    fg="white",
    activebackground="#228B22",
    activeforeground="white",
    relief="flat",
    width=20,
    command=analyze_text_input
)
analyze_btn.pack(pady=10)

# --- Result Section ---
result_label = tk.Label(root, text="Analysis Report:", font=("Segoe UI", 12, "bold"), bg="#f9f9f9", fg="black")
result_label.pack(anchor="w", padx=40, pady=(10, 5))

result_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=8, font=("Consolas", 10))
result_box.pack(padx=40, pady=5, ipady=5)
result_box.config(relief="solid", bd=1, state='disabled')

root.mainloop()

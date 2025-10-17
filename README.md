Perfect — here’s a **professional README.md** file you can directly copy-paste into your GitHub repository.
It’s complete, beginner-friendly, and lists every step, dependency, and explanation of how your system works.

---

## **MailGuard: Smart Email & Screenshot Spam Detection System**

MailGuard is a **hybrid intelligent spam detection system** that can identify spam from **typed email text** or **email screenshots (images)** using **Machine Learning** and **Optical Character Recognition (OCR)**.

It provides a clean **Graphical User Interface (GUI)** built with **Tkinter**, allowing users to interactively:

* Enter an email manually to analyze.
* Upload a screenshot of an email and extract text automatically.
* View real-time prediction results (Spam or Safe).
* Visualize model performance using `matplotlib` graphs.

---

## **📸 Features**

✅ Detect spam from **text input**
✅ Detect spam from **email screenshots** (via OCR)
✅ Real-time **result visualization** with colored feedback
✅ Clean **Tkinter GUI**
✅ Uses **TF-IDF vectorization** and **Naive Bayes model**
✅ Displays **model accuracy, precision, recall graphs**

---

## **🧠 Tech Stack & Libraries**

| Category                        | Libraries Used                           |
| ------------------------------- | ---------------------------------------- |
| **Machine Learning**            | `scikit-learn`, `pandas`, `numpy`        |
| **OCR (Image Text Extraction)** | `opencv-python`, `pytesseract`, `Pillow` |
| **Visualization**               | `matplotlib`                             |
| **GUI**                         | `tkinter` (built-in with Python)         |

---

## **⚙️ Installation & Setup**

### **1. Clone the Repository**

```bash
git clone https://github.com/anamta-JINX/MailGuard-Spam-Detection.git
cd MailGuard-Spam-Detection
```

### **2. Install Required Libraries**

Run this in your terminal (inside your project folder):

```bash
pip install pandas numpy scikit-learn opencv-python pytesseract pillow matplotlib
```

---

### **3. Install Tesseract OCR**

MailGuard uses **Tesseract OCR** to read text from images.

#### **Windows**

1. Download from: [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)
2. Install it, then note the install path (e.g. `C:\Program Files\Tesseract-OCR\tesseract.exe`)
3. Add that path to your system’s **Environment Variables → PATH**.

#### **Mac**

```bash
brew install tesseract
```

#### **Linux**

```bash
sudo apt install tesseract-ocr
```

---

### **4. Dataset**

Ensure you have the file **`spam.csv`** in your project directory.
This dataset should have at least these two columns:

```
Category, Message
```

Where:

* `Category` = “spam” or “ham”
* `Message` = actual email text

You can use the [Kaggle Spam SMS dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).

---

## **🚀 Run the Program**

### **Option 1: Run from VS Code / Terminal**

```bash
python MailGuard.py
```

### **Option 2: Double-click MailGuard.py**

If Python is installed correctly, it’ll open a graphical window automatically.

---

## **💡 How It Works**

1. **Training Phase**

   * Loads the dataset and cleans it.
   * Converts text to numerical form using **TF-IDF Vectorizer**.
   * Trains a **Naive Bayes (MultinomialNB)** classifier.

2. **Evaluation Phase**

   * Displays Accuracy, Precision, and Recall scores.
   * Generates graphs using **matplotlib**.

3. **User Interaction**

   * GUI offers two main buttons:

     * **Enter Text for Analysis** → user types an email.
     * **Upload Screenshot for Analysis** → user selects an image file.
   * The extracted or entered text is classified as **Spam (red)** or **Safe (green)**.

---

## **📊 Visualization**

The system shows two visualizations:

1. **Model Performance Graphs** — Accuracy, Precision, Recall
2. **Result Visualization** — Spam (red) / Safe (green)

---

## **🧩 Example Output**

```
Accuracy: 0.969
Precision: 1.0
Recall: 0.777
```

GUI:

* “Enter text for analysis” button → opens a text input box.
* “Upload screenshot for analysis” button → opens file picker window.
* “Analyze” button → predicts and displays color-coded results.

---

## **📦 File Structure**

```
MailGuard-Spam-Detection/
│
├── spam.csv                 # Dataset
├── MailGuard.py             # Main program
├── README.md                # Documentation (this file)
└── requirements.txt         # (Optional) dependency list
```

---

## **🧰 Optional: Create requirements.txt**

You can create a `requirements.txt` for easy dependency installation:

```bash
pip freeze > requirements.txt
```

Then in any new environment:

```bash
pip install -r requirements.txt
```

---

## **💬 Author**

**Anamta Gohar (@anamta-JINX)**
Created as part of a Machine Learning Lab project — focusing on **Human-Computer Interaction (HCI)** and **AI integration** for intelligent user experience.

---

Would you like me to include screenshots of the GUI and graphs section (with Markdown image placeholders) so your README looks visually impressive on GitHub too?

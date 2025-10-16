**MailGuard: Smart Spam Detection System**

MailGuard is a **machine learning-based spam detection system** built using the **Naive Bayes Classifier**.
It analyzes the frequency of words and phrases in email content to accurately classify messages as **Spam** or **Not Spam**.

---

🚀 Project Overview

The project demonstrates how **Natural Language Processing (NLP)** and **Machine Learning** can be combined to filter out unwanted emails efficiently.
MailGuard uses **TF-IDF vectorization** and a **Multinomial Naive Bayes model** to learn patterns from email datasets and identify common spam traits.

---

🧠 Key Features

* **Preprocessing:** Cleaned and vectorized email text using TF-IDF
* **Model:** Trained a Naive Bayes classifier for text classification
* **Evaluation Metrics:** Accuracy, Precision, and Recall
* **Visualization:** Plotted model performance metrics using Matplotlib
* **Custom Prediction:** Allows users to test with their own email text samples

---

📊 Model Performance

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 0.976 |
| Precision | 1.000 |
| Recall    | 0.831 |

MailGuard achieves a **high accuracy and perfect precision**, meaning it almost never misclassifies non-spam emails as spam.

---

🧩 Technologies Used

* **Python 3**
* **Pandas**
* **Scikit-learn**
* **Matplotlib**
* **Natural Language Processing (NLP)**

---

🗂️ Dataset

The model was trained using the **Spam Email Dataset** from Kaggle.
It contains labeled examples of spam and ham (non-spam) messages.

**Columns used:**

* `Category` → Indicates whether the message is spam or ham
* `Message` → Contains the email text content

---

⚙️ How It Works

1. **Data Loading** – Import and clean the dataset.
2. **Text Transformation** – Convert text to numerical format using TF-IDF.
3. **Training** – Fit a Multinomial Naive Bayes model.
4. **Testing** – Predict spam vs. non-spam on unseen data.
5. **Evaluation** – Compute accuracy, precision, recall, and classification report.
6. **Visualization** – Display results using bar charts for clarity.

---

🧾 Example Output

```
Accuracy: 0.976
Precision: 1.0
Recall: 0.831

Custom Email Prediction: Spam
```

---

📥 How to Run the Project

1. Clone the repository:

   ```bash
   git clone https://github.com/anamta-JINX/MailGuard-Spam-Detection.git
   ```
2. Navigate to the project folder:

   ```bash
   cd MailGuard-Spam-Detection
   ```
3. Install required dependencies:

   ```bash
   pip install pandas scikit-learn matplotlib
   ```
4. Run the script:

   ```bash
   python MailGuard.py
   ```

---

📩 Sample Test

You can modify this line in the code to test your own email:

```python
sample = ["Congratulations! You've won a free trip to Dubai. Click below to claim."]
```

Change the message and re-run the code to see how MailGuard classifies it!

---

Author:

**Anamta Gohar**
GitHub: [anamta-JINX](https://github.com/anamta-JINX)

---

Acknowledgment

Dataset Source: [Kaggle - SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

----------------------------------------------------------------------------------------------------------------------------------------------------------------


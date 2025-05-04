# Knowledge Distillation for Sentiment Classification (IMDb)

## 📌 Project Overview

This project demonstrates the process of compressing a large pretrained model into a smaller student model using **knowledge distillation**. The task is binary sentiment classification on the IMDb movie reviews dataset. The goal is to train a student model that approximates the teacher model's performance while being faster and more efficient.

---

## ▶️ How to Run

1. Open the notebook in Google Colab or Jupyter.
2. Ensure `IMDB Dataset.csv` is available in the environment.
3. Run all cells from top to bottom. The notebook automatically caches processed data, tokenizations, and models.

---

## 🧩 Steps Included in the Project

1. **Dataset Preparation**: IMDb CSV loaded, labeled, shuffled, and split into train/val/test sets.
2. **Tokenization**: Using BERT tokenizer, with results cached.
3. **Teacher Model Training**: Fine-tuned `bert-base-uncased` for 1 epoch.
4. **Soft Label Generation**: Teacher logits saved for distillation.
5. **Student Model Training**: DistilBERT trained for 1 epoch using both hard and soft labels.
6. **Evaluation**: Accuracy, F1, and AUC measured on test set.
7. **Visualization**: Bar plot comparing teacher and student performance.

---

## 🧠 Models Used

* **Teacher Model**: `BertForSequenceClassification` (BERT-base, uncased)
* **Student Model**: Custom DistilBERT-based classifier with dropout and linear head

---

## 💾 Checkpoints and Resume-ability

* All data splits, encodings, and model outputs are cached in the `checkpoints/` directory.
* If kernel/session crashes, rerun will load from cache.
* Files stored:

  * `train.pkl`, `val.pkl`, `test.pkl`: dataset splits
  * `*_encodings.pkl`: tokenized input caches
  * `soft_labels.pt`: teacher logits
  * `teacher_model/`, `student_model.pth`: trained models

> The pipeline is fully resumable and efficient for reruns.

---

## 📊 Final Results

| Model   | Accuracy | F1 Score | AUC    |
| ------- | -------- | -------- | ------ |
| Teacher | 0.9234   | 0.9256   | 0.9786 |
| Student | 0.9292   | 0.9290   | 0.9796 |


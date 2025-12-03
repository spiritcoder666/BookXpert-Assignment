# BookXpert-Assignment

## 📌 Project Overview
This repository contains the complete solution for the AI Assignment, structured into two distinct tasks. The project demonstrates the use of Vector Search for text similarity and Fine-Tuning Large Language Models (LLMs) for domain-specific tasks.

## 📂 Repository Structure
* **`task1.ipynb`**: Source code for the Name Matching System (Task 1).
* **`recipe_chatbot.ipynb`**: Source code for the Fine-Tuned Recipe LLM & Chatbot (Task 2).
* **`names_dataset.csv`**: The dataset containing name variations used in Task 1.
* **`README.md`**: Project documentation and setup guide.

---

## 🛠️ Task 1: Intelligent Name Matching
**File:** `task1.ipynb`

### Objective
To build a system that identifies and ranks similar names from a dataset, handling spelling variations (e.g., mapping "Geetha" to "Gita", "Geeta").

### Technical Approach
* **Vectorization:** Used **TF-IDF with Character N-Grams** (2-4 chars) to capture phonetic and spelling patterns.
* **Search Engine:** Implemented **FAISS (Facebook AI Similarity Search)** for high-speed dense vector retrieval.
* **Metric:** Cosine Similarity (via Inner Product of normalized vectors).

### How to Run
1.  Open `task1.ipynb` in Google Colab or Jupyter Notebook.
2.  Ensure `names_dataset.csv` is in the same directory (or uploaded to Colab).
3.  Run the cells sequentially.
4.  **Input:** Enter a name when prompted (e.g., `Geetha`).
5.  **Output:** The system returns the "Best Match" and a ranked list of "Relevant Names" with similarity scores.

---

## 🍳 Task 2: Recipe Generation Chatbot
**File:** `recipe_chatbot.ipynb`

### Objective
To fine-tune a local LLM on a recipe dataset and expose it via a Chatbot interface that suggests recipes based on user ingredients.

### Technical Approach
* **Model:** `TinyLlama-1.1B-Chat` (Optimized for efficiency on standard GPUs).
* **Fine-Tuning:** Implemented **QLoRA (Quantized Low-Rank Adaptation)** using the `peft` and `trl` libraries.
* **Training Data:** Custom JSON dataset containing Ingredient-to-Recipe mappings.
* **Interface:** A conversational loop (Chatbot) that generates step-by-step cooking instructions.

### How to Run (Google Colab Recommended)
*Note: Task 2 requires a GPU. If using Google Colab, select **Runtime > Change runtime type > T4 GPU**.*

1.  Open `recipe_chatbot.ipynb`.
2.  Run the installation cells to set up `transformers`, `peft`, `bitsandbytes`, and `trl`.
3.  Execute the **Training Cell** to fine-tune the model on the recipe data (Training takes approx. 1-2 minutes).
4.  Execute the **Inference/Chat Cell**.
5.  **Usage:**
    * **User Input:** `Egg, Onion`
    * **Bot Output:**
        ```text
        1. Chop onions and chilies.
        2. Whisk eggs with salt and pepper.
        3. Fry in oil until golden.
        ```

---

## 💻 Dependencies
To run this project locally, the following Python libraries are required:

```bash
# Task 1 Dependencies
pip install pandas numpy faiss-cpu scikit-learn

# Task 2 Dependencies (GPU Required)
pip install torch transformers peft trl bitsandbytes accelerate datasets

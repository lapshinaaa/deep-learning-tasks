# Deep Learning Tasks Repository

Welcome to my Deep Learning Tasks repository! 🚀  
This repo contains a collection of completed deep learning exercises and experiments, implemented in Jupyter Notebooks (.ipynb). Each notebook explores a different aspect of deep learning — from fundamentals to advanced architectures.  

All tasks were completed as part of the *Deep Learning course (minor: Intellectual Data Analysis)* as well as *Deep Learning-2 course* at HSE.  
Each notebook is self-contained; imports are provided inline, and main libraries are also listed in `requirements.txt`.  

---

## 📂 Repository Structure  

├── DL1_introductory_tasks.ipynb  
├── DL2_image_classification.ipynb  
├── DL3_text_classification.ipynb  
├── DL4_transformers_NER.ipynb  
├── DL5_image_segmentation.ipynb  
├── requirements.txt  
├── LICENSE <br />
└── README.md  

---

## 📑 Task Overview  

| Notebook | Topic | Key Concepts | Notes |
|----------|-------|--------------|-------|
| `DL1_introductory_tasks.ipynb` | 🔰 Introduction to Deep Learning | PyTorch basics, tensors, autograd | Introductory exercises to get familiar with PyTorch and essential DL libraries. |
| `DL2_image_classification.ipynb` | 🌱 Plant Species Classification | CNNs, custom architectures, optimization | Built and trained CNNs in PyTorch for image classification, experimented with model customization and training strategies. |
| `DL3_text_classification.ipynb` | 📰 News & Comment Classification | NLP, text classification, Hugging Face models | Trained models to classify news articles, predicted categories for unseen items, applied sentiment analysis with Hugging Face, and built analytics on most positive/negative news/comment categories. |
| `DL4_transformers_NER.ipynb` | 🏷️ Named Entity Recognition (NER) with Transformers & LLMs | Tokenizer-independent NER, BIO tagging, HuggingFace token classification, DataCollator, span alignment, LLM-assisted annotation, Optuna tuning | Built a full NER pipeline from scratch: reconstructed BIO labels into tokenizer-independent spans, aligned character-level entities to BPE tokens, and tokenized datasets for model training. Used Qwen-2.5 7B-Instruct to generate synthetic annotations, implemented strict validation, retry logic, and span post-processing, and merged valid LLM-generated samples into the training set. Fine-tuned BAAI/bge-small-en-v1.5 with HuggingFace Trainer, evaluated token-classification metrics, and achieved strong results even with limited synthetic annotation. |
| `DL5_image_segmentation.ipynb` | 🧠 Image Segmentation (U-Net, LinkNet) | Encoder–decoder architectures, skip connections, VGG backbones, loss engineering, deep supervision, post-processing, experiment tracking | Implemented U-Net and LinkNet from scratch with a VGG13 encoder. Explored architectural refinements (residual decoder blocks, batch normalization), advanced optimization strategies (BCE + Dice loss scheduling, deep supervision), and Albumentations-based data augmentation. Logged training with TensorBoard, performed systematic ablations, and improved validation IoU from baseline U-Net to 0.92+ with post-processing via morphological operations. |

---

## ⚙️ Setup & Installation  

To run the notebooks locally, you’ll need Python 3.8+ and the dependencies listed in `requirements.txt`.  

```bash
git clone https://github.com/yourusername/deep-learning-tasks.git
cd deep-learning-tasks
pip install -r requirements.txt
```

Or, open directly in Google Colab.  

---

## 🧑‍💻 Usage  

1. Launch Jupyter Notebook or Jupyter Lab:  

```bash
jupyter notebook
```

2. Open the notebook of interest (e.g., `DL1_introductory_tasks.ipynb`).  
3. Run cells step by step to explore code, results, and commentary.  

---

## 📊 Results & Visualizations  

Each notebook includes:  
 • Explanations of the approach  
 • Training/validation metrics (accuracy, loss curves)  
 • Key visualizations (e.g., confusion matrices, generated images)  
 • Reflections on results and limitations  

---

## 🛠 Dependencies  

Main libraries used across notebooks:  
 • [PyTorch](https://pytorch.org/)  
 • NumPy, Pandas  
 • Matplotlib, Seaborn  
 • scikit-learn  

See `requirements.txt` for the full list.  

---

## 🌟 Future Work  

Planned extensions:  
 • More tasks on NLP and transformers  
 • Advanced optimization and hyperparameter tuning experiments  
 • Applied case studies (healthcare, finance, etc.)  

---

## 📜 License  

This repository is released under the MIT License.  
Feel free to fork, explore, and build upon it!  

---

## 👤 Author  

Created by [Anastasiia Lapshina](https://github.com/lapshinaaa).  
Feel free to reach out via GitHub Issues if you’d like to collaborate or discuss ideas.  

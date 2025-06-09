# 📈 GNN-Based Movie Recommendation Engine (LightGCN + MovieLens)

This project implements a **Graph Neural Network (GNN)** recommendation system using the **LightGCN** architecture, built entirely in a **single Python file**, and trained on the **MovieLens 100K** dataset. The model significantly outperforms traditional baselines like Matrix Factorization and ItemKNN in all ranking metrics, making it a strong candidate for real-world personalization systems (e.g., Aniflixx).

---

## 📌 Overview

- **Model**: LightGCN (Light Graph Convolutional Network)
- **Dataset**: MovieLens 100K (explicit user ratings)
- **Frameworks**: PyTorch + PyTorch Geometric
- **Highlights**: Precision@10 ↑ 23%, NDCG@10 ↑ 21% over MF
- **Use Case**: Replacing matrix-based recommenders in sparse environments

---

## 🧠 Key Features

- ✔️ Complete implementation in **one file**
- ✔️ Graph construction using user-item interactions
- ✔️ LightGCN with BPR (Bayesian Personalized Ranking) loss
- ✔️ Dynamic negative sampling
- ✔️ Evaluation: Precision@K, Recall@K, NDCG@K, Hit Ratio, RMSE, MAP
- ✔️ Plots for training loss, NDCG progression, and ranking effectiveness

---

## 🧪 Dataset

- 📄 **MovieLens 100K**  
  > 100,000 ratings (1–5 stars) from 943 users on 1,682 movies  
  [Download](https://grouplens.org/datasets/movielens/100k/)

- ✅ Ratings ≥ 3.5 treated as **positive implicit feedback**
- 🔁 Converted to **bipartite undirected graph** (user/item nodes, rating edges)

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install torch torch-geometric numpy pandas matplotlib seaborn scikit-learn

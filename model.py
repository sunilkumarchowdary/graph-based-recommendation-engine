import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import random
import requests
import zipfile
import io
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Create output directory
output_dir = './output'
os.makedirs(output_dir, exist_ok=True)

# Ensure data directory exists
data_dir = './data/ml-100k'
os.makedirs(data_dir, exist_ok=True)

print("Step 1: Setting up the GNN-based recommendation system")

# Step 1: Download and prepare dataset if needed
def download_dataset():
    # Check if data files already exist
    if os.path.exists(os.path.join(data_dir, 'ratings.csv')):
        print("Dataset files already exist, skipping download.")
        return

    # MovieLens 100K URL
    url = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'

    print(f"Downloading MovieLens 100K dataset from {url}")

    # Download the dataset
    response = requests.get(url)

    # Extract the zip file
    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
        zip_ref.extractall(os.path.dirname(data_dir))

    # Convert to CSV format
    ratings_path = os.path.join(data_dir, 'u.data')
    if os.path.exists(ratings_path):
        ratings = pd.read_csv(
            ratings_path,
            sep='\t',
            names=['userId', 'movieId', 'rating', 'timestamp'],
            encoding='latin-1'
        )
        ratings.to_csv(os.path.join(data_dir, 'ratings.csv'), index=False)

    # Convert movies data
    movies_path = os.path.join(data_dir, 'u.item')
    if os.path.exists(movies_path):
        # Read genre information
        genres = ['unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

        movies = pd.read_csv(
            movies_path,
            sep='|',
            names=['movieId', 'title', 'release_date', 'video_release_date',
                   'IMDb_URL'] + genres,
            encoding='latin-1',
            index_col=False
        )

        # Convert genre columns to a single genres column
        movies['genres'] = ''
        for genre in genres:
            movies.loc[movies[genre] == 1, 'genres'] += genre + '|'

        # Remove trailing pipe
        movies['genres'] = movies['genres'].str.rstrip('|')

        # Select relevant columns
        movies = movies[['movieId', 'title', 'release_date', 'genres']]

        movies.to_csv(os.path.join(data_dir, 'movies.csv'), index=False)

    # Convert users data
    users_path = os.path.join(data_dir, 'u.user')
    if os.path.exists(users_path):
        users = pd.read_csv(
            users_path,
            sep='|',
            names=['userId', 'age', 'gender', 'occupation', 'zip_code'],
            encoding='latin-1'
        )
        users.to_csv(os.path.join(data_dir, 'users.csv'), index=False)

    print("Dataset downloaded and converted to CSV format")

print("Step 2: Downloading and preparing dataset")
download_dataset()

# Step 3: Implement GNN Models
print("Step 3: Implementing GNN Models")

class BaseGNNModel(nn.Module):
    """Base class for all GNN recommendation models."""

    def __init__(self, num_users, num_items, embedding_dim, num_layers, dropout=0.0):
        super(BaseGNNModel, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout

    def get_user_embeddings(self):
        """Get the learned user embeddings."""
        raise NotImplementedError

    def get_item_embeddings(self):
        """Get the learned item embeddings."""
        raise NotImplementedError

    def predict(self, user_indices, item_indices):
        """Predict the scores for user-item pairs."""
        raise NotImplementedError

    def recommend_items(self, user_indices, top_k=10, excluded_items=None):
        """Generate top-k recommendations for users."""
        raise NotImplementedError


class LightGCN(BaseGNNModel):
    """Implementation of LightGCN model for recommendation."""

    def __init__(self, num_users, num_items, embedding_dim, num_layers, dropout=0.0):
        super(LightGCN, self).__init__(num_users, num_items, embedding_dim, num_layers, dropout)

        # User and item embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # Initialize embeddings
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

    def _get_ego_embeddings(self):
        """Get the initial embeddings for all users and items."""
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self, edge_index):
        """Forward pass of the LightGCN model."""
        # Get initial embeddings
        all_embeddings = self._get_ego_embeddings()
        embeddings_list = [all_embeddings]

        # Calculate the normalization coefficient for each edge
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)

        # Calculate node degrees
        node_degrees = torch.zeros(self.num_users + self.num_items, device=edge_index.device)
        for i in range(edge_index.size(1)):
            node_degrees[edge_index[1, i]] += 1

        # Simplified approach: use a sparse matrix multiplication in message passing
        # Perform GCN propagation for each layer
        for _ in range(self.num_layers):
            # Message passing - simplified version
            temp_emb = torch.zeros_like(all_embeddings)

            # Propagate messages along each edge
            for i in range(edge_index.size(1)):
                src, dst = edge_index[0, i], edge_index[1, i]

                # Apply normalization: 1/sqrt(deg(src)) * 1/sqrt(deg(dst))
                norm = 1.0 / (torch.sqrt(node_degrees[src]) * torch.sqrt(node_degrees[dst]))
                if not torch.isfinite(norm):
                    norm = 0.0

                # Add normalized source node embedding to destination node
                temp_emb[dst] += all_embeddings[src] * norm

            # Update embeddings
            all_embeddings = temp_emb

            # Apply dropout if specified
            if self.dropout > 0:
                all_embeddings = F.dropout(all_embeddings, p=self.dropout, training=self.training)

            embeddings_list.append(all_embeddings)

        # Sum up the embeddings from each layer to get the final embeddings
        all_embeddings = torch.stack(embeddings_list, dim=1).mean(dim=1)

        # Split user and item embeddings
        user_embeddings = all_embeddings[:self.num_users]
        item_embeddings = all_embeddings[self.num_users:]

        return all_embeddings, user_embeddings, item_embeddings

    def get_user_embeddings(self):
        """Get the final user embeddings after propagation."""
        all_embeddings, user_embeddings, _ = self.forward(self.edge_index)
        return user_embeddings

    def get_item_embeddings(self):
        """Get the final item embeddings after propagation."""
        all_embeddings, _, item_embeddings = self.forward(self.edge_index)
        return item_embeddings

    def predict(self, user_indices, item_indices):
        """Predict the scores for user-item pairs."""
        # Get embeddings from the final layer
        _, user_embeddings, item_embeddings = self.forward(self.edge_index)

        # Extract the embeddings for the specified users and items
        user_emb = user_embeddings[user_indices]
        item_emb = item_embeddings[item_indices]

        # Compute the dot product of user and item embeddings
        scores = torch.sum(user_emb * item_emb, dim=1)

        return scores

    def recommend_items(self, user_indices, top_k=10, excluded_items=None):
        """Generate top-k recommendations for users."""
        # Get embeddings from the final layer
        _, user_embeddings, item_embeddings = self.forward(self.edge_index)

        # Extract user embeddings
        user_emb = user_embeddings[user_indices]

        # Calculate scores for all items
        scores = torch.matmul(user_emb, item_embeddings.t())

        # Create a mask for excluded items
        if excluded_items is not None:
            for i, user_idx in enumerate(user_indices.tolist()):
                if user_idx in excluded_items:
                    exclude_idx = torch.tensor(excluded_items[user_idx], device=scores.device)
                    scores[i].index_fill_(0, exclude_idx, float('-inf'))

        # Get top-k items
        top_scores, top_indices = torch.topk(scores, k=min(top_k, self.num_items), dim=1)

        return top_indices, top_scores

# Step 4: Create the Recommendation System class
print("Step 4: Creating Recommendation System class")

class RecommendationSystem:
    """Recommendation system based on GNN models."""

    def __init__(self, config):
        """Initialize the recommendation system."""
        self.config = config
        self.device = torch.device(config['device'])

        # Set random seeds for reproducibility
        if 'seed' in config:
            seed = config['seed']
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

        # Initialize data structures
        self.data = {}
        self.model = None
        self.edge_index = None
        self.train_history = {'train_loss': [], 'val_metrics': {}}

    def load_data(self):
        """Load data from CSV files."""
        data_dir = self.config['data_dir']

        # Load ratings data
        ratings_path = os.path.join(data_dir, 'ratings.csv')
        self.data['ratings'] = pd.read_csv(ratings_path)

        # Load movies data if available
        movies_path = os.path.join(data_dir, 'movies.csv')
        if os.path.exists(movies_path):
            self.data['movies'] = pd.read_csv(movies_path)

        # Load users data if available
        users_path = os.path.join(data_dir, 'users.csv')
        if os.path.exists(users_path):
            self.data['users'] = pd.read_csv(users_path)

        print(f"Loaded {len(self.data['ratings'])} ratings")
        if 'movies' in self.data:
            print(f"Loaded {len(self.data['movies'])} movies")
        if 'users' in self.data:
            print(f"Loaded {len(self.data['users'])} users")

    def preprocess_data(self, min_user_interactions=5, min_item_interactions=5):
        """Preprocess the data by filtering inactive users and items."""
        ratings = self.data['ratings']

        # Filter by rating threshold if specified
        if 'rating_threshold' in self.config:
            threshold = self.config['rating_threshold']
            ratings = ratings[ratings['rating'] >= threshold]

        # Count interactions per user and item
        user_counts = ratings['userId'].value_counts()
        item_counts = ratings['movieId'].value_counts()

        # Filter active users and items
        active_users = user_counts[user_counts >= min_user_interactions].index
        active_items = item_counts[item_counts >= min_item_interactions].index

        # Filter ratings to include only active users and items
        filtered_ratings = ratings[
            ratings['userId'].isin(active_users) &
            ratings['movieId'].isin(active_items)
        ]

        # Create a copy to avoid SettingWithCopyWarning
        filtered_ratings = filtered_ratings.copy()

        # Create user and item ID mappings (original ID -> index)
        unique_users = filtered_ratings['userId'].unique()
        unique_items = filtered_ratings['movieId'].unique()

        user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
        item_id_map = {iid: idx for idx, iid in enumerate(unique_items)}

        # Create reverse mappings (index -> original ID)
        self.user_index_to_id = {idx: uid for uid, idx in user_id_map.items()}
        self.item_index_to_id = {idx: iid for iid, idx in item_id_map.items()}

        # Update the ratings with mapped indices
        filtered_ratings.loc[:, 'user_idx'] = filtered_ratings['userId'].map(user_id_map)
        filtered_ratings.loc[:, 'item_idx'] = filtered_ratings['movieId'].map(item_id_map)

        # Store the preprocessed data
        self.data['preprocessed_ratings'] = filtered_ratings
        self.data['user_id_map'] = user_id_map
        self.data['item_id_map'] = item_id_map

        # Store the number of users and items for model initialization
        self.num_users = len(unique_users)
        self.num_items = len(unique_items)

        print(f"After preprocessing: {len(filtered_ratings)} ratings, "
              f"{self.num_users} users, {self.num_items} items")

    def split_data(self, test_size=0.2, val_size=0.1):
        """Split data into training, validation, and test sets."""
        ratings = self.data['preprocessed_ratings']

        # First split into training+validation and test
        train_val_data, test_data = train_test_split(
            ratings, test_size=test_size, stratify=ratings['user_idx'], random_state=self.config.get('seed', 42)
        )

        # Then split training+validation into training and validation
        if val_size > 0:
            train_data, val_data = train_test_split(
                train_val_data, test_size=val_size, stratify=train_val_data['user_idx'],
                random_state=self.config.get('seed', 42)
            )
        else:
            train_data = train_val_data
            val_data = None

        # Store the splits
        self.data['train_data'] = train_data
        self.data['val_data'] = val_data
        self.data['test_data'] = test_data

        # Create sets of user-item interactions for fast lookup during negative sampling
        self.user_interactions = defaultdict(set)
        for _, row in train_data.iterrows():
            self.user_interactions[row['user_idx']].add(row['item_idx'])

        print(f"Data split: {len(train_data)} train, "
              f"{len(val_data) if val_data is not None else 0} validation, "
              f"{len(test_data)} test")

    def create_graph(self):
        """Create a bipartite graph from user-item interactions."""
        train_data = self.data['train_data']

        # Create user-item interaction edges
        edges = []

        # User -> Item edges
        for _, row in train_data.iterrows():
            user_idx = row['user_idx']
            item_idx = row['item_idx'] + self.num_users  # Offset item indices
            edges.append([user_idx, item_idx])
            edges.append([item_idx, user_idx])  # Add reverse edge for undirected graph

        # Convert to tensor
        edge_index = torch.tensor(edges, dtype=torch.long, device=self.device).t()

        self.edge_index = edge_index

        print(f"Created graph with {self.edge_index.size(1)} edges")
        return self.edge_index

    def initialize_model(self):
        """Initialize the GNN model."""
        model_type = self.config['model_type'].lower()

        if model_type == 'lightgcn':
            self.model = LightGCN(
                num_users=self.num_users,
                num_items=self.num_items,
                embedding_dim=self.config['embedding_dim'],
                num_layers=self.config['num_layers'],
                dropout=self.config.get('dropout', 0.0)
            ).to(self.device)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Store the edge index in the model for easy access
        self.model.edge_index = self.edge_index

        print(f"Initialized {model_type.upper()} model with "
              f"{sum(p.numel() for p in self.model.parameters())} parameters")

    def sample_negative_items(self, user_idx, n_neg=1):
        """Sample negative items for a user."""
        positive_items = self.user_interactions[user_idx]

        neg_items = []
        while len(neg_items) < n_neg:
            neg_item = random.randint(0, self.num_items - 1)
            if neg_item not in positive_items and neg_item not in neg_items:
                neg_items.append(neg_item)

        return neg_items

    def bpr_loss(self, pos_scores, neg_scores):
        """Compute the Bayesian Personalized Ranking (BPR) loss."""
        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10))
        return loss

    def train_epoch(self, optimizer, neg_samples=1, batch_size=1024):
        """Train the model for one epoch."""
        self.model.train()
        total_loss = 0.0
        train_data = self.data['train_data']

         # Create batches of user-item interactions
        n_batches = max(1, (len(train_data) + batch_size - 1) // batch_size)

         # Make sure we're using indices that are valid
        valid_indices = train_data.index.to_list()
        batches = np.array_split(valid_indices, n_batches)
        for batch_indices in tqdm(batches, desc="Training", leave=False):
            optimizer.zero_grad()
        # Ensure indices are valid
            valid_batch_indices = [idx for idx in batch_indices if idx in train_data.index]
            if not valid_batch_indices:
                continue  # Skip this batch if no valid indices

            batch = train_data.loc[valid_batch_indices]
            user_indices = torch.tensor(batch['user_idx'].values, dtype=torch.long, device=self.device)
            pos_item_indices = torch.tensor(batch['item_idx'].values, dtype=torch.long, device=self.device)
            # Sample negative items
            neg_item_indices_list = []
            for user_idx in batch['user_idx'].values:
                neg_items = self.sample_negative_items(user_idx, neg_samples)
                neg_item_indices_list.extend(neg_items)

            neg_item_indices = torch.tensor(neg_item_indices_list, dtype=torch.long, device=self.device)

            # Repeat user indices for multiple negative samples
            if neg_samples > 1:
                user_indices = user_indices.repeat_interleave(neg_samples)

            # Forward pass
            _, user_embeddings, item_embeddings = self.model.forward(self.edge_index)

            # Get embeddings for positive and negative items
            users_emb = user_embeddings[user_indices]
            pos_items_emb = item_embeddings[pos_item_indices]
            neg_items_emb = item_embeddings[neg_item_indices]

            # Calculate scores
            pos_scores = torch.sum(users_emb * pos_items_emb, dim=1)
            neg_scores = torch.sum(users_emb * neg_items_emb, dim=1)

            # Compute loss
            loss = self.bpr_loss(pos_scores, neg_scores)

            # Add L2 regularization if specified
            if 'weight_decay' in self.config and self.config['weight_decay'] > 0:
                l2_reg = 0
                for param in self.model.parameters():
                    l2_reg += torch.norm(param, 2)
                loss += self.config['weight_decay'] * l2_reg

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / n_batches
        return avg_loss

    def evaluate(self, data_split='val', k_values=[5, 10, 20], batch_size=1024):
        """Evaluate the model on validation or test data."""
        self.model.eval()

        if data_split == 'val':
            eval_data = self.data['val_data']
        elif data_split == 'test':
            eval_data = self.data['test_data']
        else:
            raise ValueError(f"Invalid data split: {data_split}")

        # Initialize metrics
        metrics = {
            'precision': {k: 0.0 for k in k_values},
            'recall': {k: 0.0 for k in k_values},
            'ndcg': {k: 0.0 for k in k_values},
            'hit_ratio': {k: 0.0 for k in k_values}
        }

        # Create user-item test interactions dictionary
        user_test_items = defaultdict(list)
        for _, row in eval_data.iterrows():
            user_test_items[row['user_idx']].append(row['item_idx'])

        # Get all users with test items
        test_users = list(user_test_items.keys())

        # Process in batches
        n_batches = (len(test_users) + batch_size - 1) // batch_size
        user_batches = np.array_split(test_users, n_batches)

        with torch.no_grad():
            for user_batch in tqdm(user_batches, desc=f"Evaluating ({data_split})", leave=False):
                user_indices = torch.tensor(user_batch, dtype=torch.long, device=self.device)

                # Get excluded items (items the user has already interacted with in training)
                excluded_items = {}
                for user_idx in user_batch:
                    excluded_items[user_idx] = list(self.user_interactions[user_idx])

                # Generate recommendations
                top_indices, top_scores = self.model.recommend_items(
                    user_indices, top_k=max(k_values), excluded_items=excluded_items
                )

                # Convert to CPU for metric calculation
                top_indices = top_indices.cpu().numpy()

                # Calculate metrics for each user
                for i, user_idx in enumerate(user_batch):
                    true_items = user_test_items[user_idx]
                    recommended_items = top_indices[i]

                    # Calculate metrics for different k values
                    for k in k_values:
                        # Get top-k recommendations
                        top_k_items = recommended_items[:k]

                        # Calculate precision
                        precision = len(set(top_k_items) & set(true_items)) / k
                        metrics['precision'][k] += precision

                        # Calculate recall
                        recall = len(set(top_k_items) & set(true_items)) / len(true_items) if true_items else 0
                        metrics['recall'][k] += recall

                        # Calculate hit ratio
                        hit_ratio = 1.0 if len(set(top_k_items) & set(true_items)) > 0 else 0.0
                        metrics['hit_ratio'][k] += hit_ratio

                        # Calculate NDCG
                        dcg = 0.0
                        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(true_items), k)))

                        for j, item in enumerate(top_k_items):
                            if item in true_items:
                                dcg += 1.0 / np.log2(j + 2)

                        ndcg = dcg / idcg if idcg > 0 else 0
                        metrics['ndcg'][k] += ndcg

        # Calculate average metrics across all users
        num_users = len(test_users)
        for metric in metrics:
            for k in k_values:
                metrics[metric][k] /= num_users

        return metrics

    def train(self):
        """Train the model using BPR loss with early stopping."""
        # Initialize optimizer
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=0  # L2 regularization is handled separately in train_epoch
        )

        # Initialize training parameters
        neg_samples = self.config.get('neg_samples', 1)
        batch_size = self.config.get('batch_size', 1024)
        num_epochs = self.config.get('num_epochs', 100)
        patience = self.config.get('early_stopping_patience', 10)

        # Initialize early stopping variables
        best_metric = 0.0
        best_epoch = 0
        no_improvement = 0
        best_model_state = None

        # Main training loop
        for epoch in range(1, num_epochs + 1):
            # Train for one epoch
            train_loss = self.train_epoch(optimizer, neg_samples, batch_size)
            self.train_history['train_loss'].append(train_loss)

            # Evaluate on validation set
            if self.data['val_data'] is not None:
                val_metrics = self.evaluate('val', k_values=[10])

                # Store validation metrics
                for metric, values in val_metrics.items():
                    if metric not in self.train_history['val_metrics']:
                        self.train_history['val_metrics'][metric] = {}

                    for k, value in values.items():
                        key = f"{metric}@{k}"
                        if key not in self.train_history['val_metrics']:
                            self.train_history['val_metrics'][key] = []
                        self.train_history['val_metrics'][key].append(value)

                # Use NDCG@10 as the early stopping metric
                current_metric = val_metrics['ndcg'][10]

                # Print epoch results
                print(f"Epoch {epoch}/{num_epochs}, Loss: {train_loss:.4f}, "
                      f"NDCG@10: {current_metric:.4f}, "
                      f"Precision@10: {val_metrics['precision'][10]:.4f}, "
                      f"Recall@10: {val_metrics['recall'][10]:.4f}")

                # Check for improvement
                if current_metric > best_metric:
                    best_metric = current_metric
                    best_epoch = epoch
                    no_improvement = 0

                    # Save the best model
                    best_model_state = {
                        'model_state_dict': self.model.state_dict(),
                        'epoch': epoch,
                        'metrics': val_metrics
                    }
                else:
                    no_improvement += 1
                    print(f"No improvement for {no_improvement} epochs "
                          f"(best NDCG@10: {best_metric:.4f} at epoch {best_epoch})")

                # Early stopping
                if no_improvement >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            else:
                # If no validation data, just print the training loss
                print(f"Epoch {epoch}/{num_epochs}, Loss: {train_loss:.4f}")

        # Load the best model if validation was used
        if self.data['val_data'] is not None and best_model_state is not None:
            self.model.load_state_dict(best_model_state['model_state_dict'])
            print(f"Loaded best model from epoch {best_model_state['epoch']}")

        return self.train_history

    def generate_recommendations(self, user_id, top_k=10):
        """Generate recommendations for a specific user."""
        self.model.eval()

        # Convert user ID to index
        if user_id not in self.data['user_id_map']:
            raise ValueError(f"User ID {user_id} not found in the dataset")

        user_idx = self.data['user_id_map'][user_id]

        # Get items the user has already interacted with
        excluded_items = {user_idx: list(self.user_interactions[user_idx])}

        # Generate recommendations
        with torch.no_grad():
            user_indices = torch.tensor([user_idx], dtype=torch.long, device=self.device)
            top_indices, top_scores = self.model.recommend_items(
                user_indices, top_k=top_k, excluded_items=excluded_items
            )

        # Convert item indices back to original IDs
        top_indices = top_indices.cpu().numpy()[0]
        top_scores = top_scores.cpu().numpy()[0]

        recommended_items = [self.item_index_to_id[idx] for idx in top_indices]

        return recommended_items, top_scores

    def get_item_details(self, item_ids):
        """Get details for a list of item IDs."""
        if 'movies' not in self.data:
            return None

        # Filter movies that exist in the dataset
        valid_item_ids = [item_id for item_id in item_ids if item_id in self.data['movies']['movieId'].values]
        if not valid_item_ids:
            return pd.DataFrame()

        return self.data['movies'][self.data['movies']['movieId'].isin(valid_item_ids)]

# Step 5: Add Analysis and Visualization Features
print("Step 5: Adding Analysis and Visualization Features")

class AnalysisAndVisualization:
    """Class for analyzing and visualizing the recommendation system results."""

    def __init__(self, recommendation_system):
        """Initialize the analysis and visualization class."""
        self.recommendation_system = recommendation_system
        self.model = recommendation_system.model
        self.data = recommendation_system.data
        self.config = recommendation_system.config
        self.device = recommendation_system.device

        # Create output directory for plots
        self.plots_dir = os.path.join(output_dir, 'plots')
        os.makedirs(self.plots_dir, exist_ok=True)

    def visualize_embeddings(self, n_users=50, n_items=50, method='tsne'):
        """Visualize user and item embeddings using dimensionality reduction."""
        if self.model is None:
            print("Model not available for embedding visualization")
            return None

        # Get user and item embeddings
        with torch.no_grad():
            user_embeddings = self.model.get_user_embeddings().cpu().numpy()
            item_embeddings = self.model.get_item_embeddings().cpu().numpy()

        # Sample users and items
        user_indices = np.random.choice(
            user_embeddings.shape[0],
            min(n_users, user_embeddings.shape[0]),
            replace=False
        )
        item_indices = np.random.choice(
            item_embeddings.shape[0],
            min(n_items, item_embeddings.shape[0]),
            replace=False
        )

        # Get sampled embeddings
        sampled_user_emb = user_embeddings[user_indices]
        sampled_item_emb = item_embeddings[item_indices]

        # Combine embeddings for dimensionality reduction
        combined_emb = np.vstack([sampled_user_emb, sampled_item_emb])

        # Apply dimensionality reduction
        if method.lower() == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
        else:  # Default to PCA
            reducer = PCA(n_components=2, random_state=42)

        reduced_emb = reducer.fit_transform(combined_emb)

        # Split reduced embeddings back into user and item parts
        reduced_user_emb = reduced_emb[:len(sampled_user_emb)]
        reduced_item_emb = reduced_emb[len(sampled_user_emb):]

        # Create figure
        plt.figure(figsize=(10, 8))

        # Plot users and items
        plt.scatter(reduced_user_emb[:, 0], reduced_user_emb[:, 1], c='blue', marker='o', alpha=0.7, label='Users')
        plt.scatter(reduced_item_emb[:, 0], reduced_item_emb[:, 1], c='red', marker='x', alpha=0.7, label='Items')

        # Add title and labels
        plt.title(f'User and Item Embeddings ({method.upper()})')
        plt.xlabel(f'{method.upper()} Dimension 1')
        plt.ylabel(f'{method.upper()} Dimension 2')
        plt.grid(alpha=0.3)
        plt.legend()

        # Save figure
        plt.savefig(os.path.join(self.plots_dir, f'embeddings_{method}.png'), dpi=300)
        plt.close()

        return reduced_user_emb, reduced_item_emb

    def plot_training_history(self):
        """Plot the training history."""
        train_history = self.recommendation_system.train_history

        if not train_history['train_loss']:
            print("No training history available to plot")
            return None

        # Create figure
        plt.figure(figsize=(12, 5))

        # Plot 1: Training Loss
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(train_history['train_loss']) + 1), train_history['train_loss'], marker='o')
        plt.title('Training Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(alpha=0.3)

        # Plot 2: Validation Metrics if available
        if train_history['val_metrics']:
            plt.subplot(1, 2, 2)

            for metric_name, values_dict in train_history['val_metrics'].items():
                if isinstance(values_dict, dict):
                    for k, values in values_dict.items():
                        if isinstance(values, list):
                            plt.plot(range(1, len(values) + 1), values, marker='o', label=f'{metric_name}@{k}')
                elif isinstance(values_dict, list):
                    plt.plot(range(1, len(values_dict) + 1), values_dict, marker='o', label=metric_name)

            plt.title('Validation Metrics over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Metric Value')
            plt.grid(alpha=0.3)
            plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'training_history.png'), dpi=300)
        plt.close()

        return train_history

    def analyze_user_preferences(self, user_id=None, n_users=5):
        """Analyze user preferences based on interactions and recommendations."""
        if 'movies' not in self.data:
            print("Movie metadata not available for preference analysis")
            return None

        # Select users to analyze
        if user_id is not None:
            if user_id not in self.recommendation_system.data['user_id_map']:
                print(f"User ID {user_id} not found in the dataset")
                return None
            user_ids = [user_id]
        else:
            # Randomly select n_users
            user_indices = np.random.choice(
                list(self.recommendation_system.user_index_to_id.keys()),
                size=min(n_users, self.recommendation_system.num_users),
                replace=False
            )
            user_ids = [self.recommendation_system.user_index_to_id[idx] for idx in user_indices]

        user_preferences = {}

        for user_id in user_ids:
            try:
                # Get user index
                user_idx = self.recommendation_system.data['user_id_map'][user_id]

                # Get items the user has interacted with
                train_data = self.data['train_data']
                user_items = train_data[train_data['user_idx'] == user_idx]['item_idx'].values
                user_item_ids = [self.recommendation_system.item_index_to_id[idx] for idx in user_items]

                # Get recommendations for the user
                rec_items, rec_scores = self.recommendation_system.generate_recommendations(user_id, top_k=10)

                # Get movie details
                interacted_movies = self.recommendation_system.get_item_details(user_item_ids)
                recommended_movies = self.recommendation_system.get_item_details(rec_items)

                # Get genre distributions
                interacted_genres = self._get_genre_distribution(interacted_movies)
                recommended_genres = self._get_genre_distribution(recommended_movies)

                # Store user preferences
                user_preferences[user_id] = {
                    'interaction_count': len(user_items),
                    'interacted_items': user_item_ids,
                    'recommended_items': rec_items,
                    'recommendation_scores': rec_scores.tolist(),
                    'interacted_genres': interacted_genres,
                    'recommended_genres': recommended_genres
                }

                # Visualize user preferences
                self._visualize_user_preferences(user_id, user_preferences[user_id], interacted_movies, recommended_movies)

            except Exception as e:
                print(f"Error analyzing preferences for user {user_id}: {str(e)}")

        return user_preferences

    def _get_genre_distribution(self, movies_df):
        """Get genre distribution from a DataFrame of movies."""
        genres_counter = Counter()

        if movies_df is None or movies_df.empty:
            return {}

        for _, row in movies_df.iterrows():
            if 'genres' in row:
                genres = row['genres'].split('|')
                genres_counter.update(genres)

        return dict(genres_counter)

    def _visualize_user_preferences(self, user_id, user_data, interacted_movies, recommended_movies):
        """Visualize user preferences."""
        plt.figure(figsize=(15, 10))

        # Plot 1: Genre distributions
        plt.subplot(2, 1, 1)

        # Combine all genres from interactions and recommendations
        all_genres = set(user_data['interacted_genres'].keys()) | set(user_data['recommended_genres'].keys())

        if not all_genres:
            plt.text(0.5, 0.5, "No genre data available", ha='center', va='center')
        else:
            # Sort genres by total count
            genre_counts = {}
            for genre in all_genres:
                interacted_count = user_data['interacted_genres'].get(genre, 0)
                recommended_count = user_data['recommended_genres'].get(genre, 0)
                genre_counts[genre] = interacted_count + recommended_count

            # Get top genres
            top_genres = [genre for genre, _ in sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:10]]

            if not top_genres:
                plt.text(0.5, 0.5, "No top genres found", ha='center', va='center')
            else:
                # Prepare data for plotting
                x = np.arange(len(top_genres))
                interacted_counts = [user_data['interacted_genres'].get(genre, 0) for genre in top_genres]
                recommended_counts = [user_data['recommended_genres'].get(genre, 0) for genre in top_genres]

                # Plot bars
                width = 0.35
                plt.bar(x - width/2, interacted_counts, width, label='Interacted')
                plt.bar(x + width/2, recommended_counts, width, label='Recommended')

                plt.xlabel('Genre')
                plt.ylabel('Count')
                plt.title(f'User {user_id}: Genre Distribution')
                plt.xticks(x, top_genres, rotation=45, ha='right')
                plt.legend()

        # Plot 2: Recommendation scores
        plt.subplot(2, 1, 2)

        # Get movie titles
        rec_items = user_data['recommended_items']
        rec_scores = user_data['recommendation_scores']
        movie_titles = []

        for i, item_id in enumerate(rec_items):
            title = f"Item {item_id}"
            if recommended_movies is not None and not recommended_movies.empty:
                item_data = recommended_movies[recommended_movies['movieId'] == item_id]
                if not item_data.empty and 'title' in item_data.columns:
                    title = item_data['title'].values[0]
                    if len(title) > 30:
                        title = title[:27] + '...'
            movie_titles.append(title)

        if not movie_titles:
            plt.text(0.5, 0.5, "No recommendations available", ha='center', va='center')
        else:
            # Plot horizontal bar chart
            plt.barh(range(len(rec_scores)), rec_scores, align='center')
            plt.yticks(range(len(rec_scores)), movie_titles)
            plt.xlabel('Score')
            plt.title(f'User {user_id}: Top Recommendations')

        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f'user_{user_id}_preferences.png'), dpi=300)
        plt.close()

# Step 6: Create the main function
print("Step 6: Creating main function and running the recommendation system")

def run_recommendation_system():
    """Run the GNN-based recommendation system."""
    # Configuration
    config = {
        'data_dir': data_dir,
        'output_dir': output_dir,
        'model_type': 'lightgcn',
        'embedding_dim': 64,
        'num_layers': 2,  # Reduced for faster execution
        'learning_rate': 0.001,
        'batch_size': 1024,
        'num_epochs': 10,  # Reduced for faster execution
        'neg_samples': 1,
        'weight_decay': 0.0001,
        'early_stopping_patience': 3,
        'min_user_interactions': 5,
        'min_item_interactions': 5,
        'test_size': 0.2,
        'val_size': 0.1,
        'rating_threshold': 3.5,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seed': 42
    }

    # Step 1: Create the recommendation system
    recommendation_system = RecommendationSystem(config)

    # Step 2: Load and preprocess data
    recommendation_system.load_data()
    recommendation_system.preprocess_data()
    recommendation_system.split_data()

    # Step 3: Create the interaction graph
    recommendation_system.create_graph()

    # Step 4: Initialize the model
    recommendation_system.initialize_model()

    # Step 5: Train the model
    print("\nTraining the model...")
    train_history = recommendation_system.train()

    # Step 6: Evaluate the model
    print("\nEvaluating the model...")
    test_metrics = recommendation_system.evaluate('test', k_values=[5, 10, 20])

    # Print evaluation results
    print("\nTest Metrics:")
    for metric in test_metrics:
        for k, value in test_metrics[metric].items():
            print(f"  {metric.upper()}@{k}: {value:.4f}")

    # Step 7: Generate recommendations for sample users
    print("\nGenerating recommendations for sample users...")

    # Get sample users
    sample_user_indices = np.random.choice(
        list(recommendation_system.user_index_to_id.keys()),
        size=min(5, recommendation_system.num_users),
        replace=False
    )
    sample_user_ids = [recommendation_system.user_index_to_id[idx] for idx in sample_user_indices]

    # Generate and print recommendations
    for user_id in sample_user_ids:
        print(f"\nRecommendations for User {user_id}:")
        try:
            rec_items, rec_scores = recommendation_system.generate_recommendations(user_id, top_k=5)

            # Get movie titles if available
            if 'movies' in recommendation_system.data:
                for i, (item_id, score) in enumerate(zip(rec_items, rec_scores)):
                    item_data = recommendation_system.data['movies'][recommendation_system.data['movies']['movieId'] == item_id]
                    if not item_data.empty:
                        title = item_data['title'].values[0]
                        print(f"  {i+1}. {title} (Score: {score:.4f})")
                    else:
                        print(f"  {i+1}. Item {item_id} (Score: {score:.4f})")
            else:
                for i, (item_id, score) in enumerate(zip(rec_items, rec_scores)):
                    print(f"  {i+1}. Item {item_id} (Score: {score:.4f})")
        except Exception as e:
            print(f"  Error generating recommendations: {str(e)}")

    # Step 8: Analyze and visualize results
    print("\nAnalyzing and visualizing results...")
    analysis = AnalysisAndVisualization(recommendation_system)

    # Visualize embeddings
    print("Visualizing embeddings...")
    analysis.visualize_embeddings(n_users=50, n_items=50)

    # Plot training history
    print("Plotting training history...")
    analysis.plot_training_history()

    # Analyze user preferences
    print("Analyzing user preferences...")
    for user_id in sample_user_ids[:2]:  # Analyze first 2 sample users
        analysis.analyze_user_preferences(user_id=user_id)

    print("\nRecommendation system completed successfully!")
    print(f"Results and visualizations saved to {output_dir}")

    return {
        'recommendation_system': recommendation_system,
        'test_metrics': test_metrics,
        'train_history': train_history
    }

# Run the system
try:
    results = run_recommendation_system()

    # Display a summary of the results
    print("\n===== SUMMARY =====")
    print("Model: LightGCN")
    print(f"Embedding dimension: {results['recommendation_system'].config['embedding_dim']}")
    print(f"Number of layers: {results['recommendation_system'].config['num_layers']}")
    print("\nTest Metrics:")
    for metric in results['test_metrics']:
        for k, value in results['test_metrics'][metric].items():
            print(f"  {metric.upper()}@{k}: {value:.4f}")

    # Show paths to visualization files
    print("\nVisualization files:")
    for root, dirs, files in os.walk(os.path.join(output_dir, 'plots')):
        for file in files:
            if file.endswith('.png'):
                print(f"  {os.path.join(root, file)}")
except Exception as e:
    print(f"An error occurred: {str(e)}")
    import traceback
    traceback.print_exc()

import os
import polars as pl
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_distances
import numpy as np
from tqdm import tqdm
import pickle
from sklearn.cluster import DBSCAN

class EssayDataset:
    def __init__(self, main_file, feature_file, readability_file):
        self.main_data = pl.read_excel(main_file)
        self.feature_data = pl.read_csv(feature_file)
        self.readability_data = pl.read_csv(readability_file)

        # 得点のスケーリング用の範囲
        self.score_ranges = {
            1: {'min': 2, 'max': 12},
            2: {'min': 1, 'max': 6},
            3: {'min': 0, 'max': 3},
            4: {'min': 0, 'max': 3},
            5: {'min': 0, 'max': 4},
            6: {'min': 0, 'max': 4},
            7: {'min': 0, 'max': 30},
            8: {'min': 0, 'max': 60},
        }

    def preprocess_dataframe(self):
        ##############################
        # メインデータの前処理
        ##############################
        self.main_data = self.main_data.drop_nulls('domain1_score') # プロンプト4に得点がないデータが存在するので削除
        self.main_data = self.main_data.rename({'domain1_score': 'original_score'})
        # スケーリング関数の定義
        def scale_score(score, essay_set):
            min_score = self.score_ranges[essay_set]['min']
            max_score = self.score_ranges[essay_set]['max']
            return (score - min_score) / (max_score - min_score)
        # domain1_scoreのスケーリング
        self.main_data = self.main_data.with_columns(
            pl.struct(['original_score', 'essay_set'])
            .map_elements(lambda x: scale_score(x['original_score'], x['essay_set']), return_dtype=pl.Float64)
            .alias('scaled_score')
        )
        self.main_data = self.main_data.select(['essay_id', 'essay_set', 'essay', 'original_score', 'scaled_score'])
        
        ##############################
        # 特徴量データの前処理
        ##############################
        self.feature_data = self.feature_data.rename({'item_id': 'essay_id', 'prompt_id': 'essay_set'})
        self.feature_data = self.feature_data.drop('score')

        # essay_set単位でmin-maxスケーリングを適用し、最後に結合
        scaled_feature_data_list = []

        for essay_set in self.feature_data['essay_set'].unique():
            # 各essay_setごとにフィルタリング
            set_data = self.feature_data.filter(pl.col('essay_set') == essay_set)

            # 3列目以降に対してmin-maxスケーリングを適用
            for col in set_data.columns[2:]:  # essay_idとessay_set以外の列
                min_val = set_data[col].min()
                max_val = set_data[col].max()

                # スケーリングを適用
                set_data = set_data.with_columns(
                    ((pl.col(col) - min_val) / (max_val - min_val)).alias(col)
                )

            # スケーリング済みのデータをリストに追加
            scaled_feature_data_list.append(set_data)

        # スケーリング済みデータを結合
        self.feature_data = pl.concat(scaled_feature_data_list)
        
        ##############################
        # 読みやすさデータの前処理
        ##############################
        self.readability_data = self.readability_data.rename({'item_id': 'essay_id'})

    def prompt_specific_split(self, test_essay_set, test_size=0.2, dev_size=0.1, random_state=42):
        # essay_idを使用してデータを分割
        essay_ids = self.main_data.filter(pl.col('essay_set') == test_essay_set)['essay_id'].to_list()
        train_ids, test_ids = train_test_split(
            essay_ids, test_size=test_size, random_state=random_state, shuffle=True
        )
        train_ids, dev_ids = train_test_split(
            train_ids, test_size=dev_size / (1 - test_size), random_state=random_state, shuffle=True
        )

        # データ抽出のためのヘルパー関数を定義
        def extract_data(ids):
            main = self.main_data.filter(pl.col('essay_id').is_in(ids))
            feature = self.feature_data.filter(pl.col('essay_id').is_in(ids))
            readability = self.readability_data.filter(pl.col('essay_id').is_in(ids))
            return {
                'essay_id': main['essay_id'].to_list(),
                'essay_set': main['essay_set'].to_list(),
                'essay': main['essay'].to_list(),
                'original_score': main['original_score'].to_list(),
                'scaled_score': main['scaled_score'].to_list(),
                'feature': feature.to_numpy(),
                'readability': readability.to_numpy(),
            }

        # 各データセットを作成
        train_data = extract_data(train_ids)
        dev_data = extract_data(dev_ids)
        test_data = extract_data(test_ids)

        return train_data, dev_data, test_data
    
    def cross_prompt_split(self, target_prompt_set, dev_size=30, cache_dir='.embedding_cache'):
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Filter data into source and target datasets
        source_data = self.main_data.filter(pl.col('essay_set') != target_prompt_set)
        target_data = self.main_data.filter(pl.col('essay_set') == target_prompt_set)
        
        # Load BERT tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        model = AutoModel.from_pretrained('bert-base-uncased')
        
        # Function to compute embeddings and cache them
        def compute_embeddings(df, cache_dir, tokenizer, model):
            embeddings_dict = {}
            uncached_essays = []
            uncached_ids = []
            
            for essay_id, essay in zip(df['essay_id'], df['essay']):
                cache_file = os.path.join(cache_dir, f'{essay_id}.pkl')
                if os.path.exists(cache_file):
                    with open(cache_file, 'rb') as f:
                        embedding = pickle.load(f)
                    embeddings_dict[essay_id] = embedding
                else:
                    uncached_essays.append(essay)
                    uncached_ids.append(essay_id)
            
            # Compute embeddings for uncached essays
            batch_size = 32
            for i in tqdm(range(0, len(uncached_essays), batch_size)):
                batch_essays = uncached_essays[i:i+batch_size]
                inputs = tokenizer(batch_essays, return_tensors='pt', padding=True, truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
                
                # Save embeddings to cache and assign to embeddings_dict
                for essay_id, embedding in zip(uncached_ids[i:i+batch_size], batch_embeddings):
                    embeddings_dict[essay_id] = embedding
                    cache_file = os.path.join(cache_dir, f'{essay_id}.pkl')
                    with open(cache_file, 'wb') as f:
                        pickle.dump(embedding, f)
            
            return embeddings_dict
        
        # Compute embeddings for source and target datasets
        source_embeddings_dict = compute_embeddings(source_data, cache_dir, tokenizer, model)
        target_embeddings_dict = compute_embeddings(target_data, cache_dir, tokenizer, model)
        all_embeddings_dict = source_embeddings_dict | target_embeddings_dict
        
        target_ids = target_data['essay_id'].to_list()
        target_embeddings_array = np.vstack([target_embeddings_dict[essay_id] for essay_id in target_ids])
        
        # Function to select diverse samples for the dev set
        def select_diverse_samples(embeddings, n_samples):
            selected_indices = []
            remaining_indices = list(range(len(embeddings)))

            # Select the first sample randomly
            first_index = np.random.choice(remaining_indices)
            selected_indices.append(first_index)
            remaining_indices.remove(first_index)

            for _ in range(1, n_samples):
                distances = cosine_distances(embeddings[selected_indices], embeddings[remaining_indices])
                sum_distances = distances.sum(axis=0)
                next_index = remaining_indices[np.argmax(sum_distances)]
                selected_indices.append(next_index)
                remaining_indices.remove(next_index)

            return selected_indices

        # Select indices for the dev set
        dev_indices = select_diverse_samples(target_embeddings_array, dev_size)
        dev_indices = [int(i) for i in dev_indices]
        # Get IDs and embeddings for dev and test sets
        test_indices = [i for i in range(len(target_ids)) if i not in dev_indices]
        
        # Function to extract data and include embeddings
        def extract_data(df, embeddings_dict):
            # Sort feature and readability DataFrames by essay_id
            df = df.sort('essay_id')
            feature = self.feature_data.filter(pl.col('essay_id').is_in(df['essay_id']))
            readability = self.readability_data.filter(pl.col('essay_id').is_in(df['essay_id']))
            embeddings = np.vstack([embeddings_dict[essay_id] for essay_id in df['essay_id']])

            # # Print for verification
            # print(df['essay_id'].to_list())
            # print(feature['essay_id'].to_list())
            # print(readability['essay_id'].to_list())

            # Assertions to ensure data integrity
            assert len(df) == len(feature) == len(readability) == embeddings.shape[0], "Data lengths must match."
            assert df['essay_id'].to_list() == feature['essay_id'].to_list(), "Essay IDs must match."
            assert df['essay_id'].to_list() == readability['essay_id'].to_list(), "Essay IDs must match."

            return {
                'essay_id': df['essay_id'].to_numpy(),
                'essay_set': df['essay_set'].to_numpy(),
                'essay': df['essay'].to_numpy(),
                'original_score': df['original_score'].to_numpy(),
                'scaled_score': df['scaled_score'].to_numpy(),
                'feature': feature.drop(['essay_id', 'essay_set']).to_numpy(),
                'readability': readability.drop(['essay_id']).to_numpy(),
                'embedding': embeddings,
            }

        # Prepare DataFrames for train, dev, and test datasets
        train_data_df = source_data
        dev_data_df = target_data[dev_indices]
        test_data_df = target_data[test_indices]

        # Extract data and include embeddings
        train_data = extract_data(train_data_df, all_embeddings_dict)
        dev_data = extract_data(dev_data_df, all_embeddings_dict)
        test_data = extract_data(test_data_df, all_embeddings_dict)

        return train_data, dev_data, test_data
    
    def clustering(self, embedding_data, prompt_data, label_data):
        # クラスタリングの割り当てを計算
        cluster_assignments = np.full(len(embedding_data), -1, dtype=int)  # -1で初期化
        cluster_centroids = np.zeros_like(embedding_data)  # x_sourceと同じ形状

        unique_prompts = np.unique(prompt_data)
        unique_labels = np.unique(label_data)

        cluster_id_counter = 0  # クラスタIDをユニークにするためのカウンター

        for prompt in unique_prompts:
            for label in unique_labels:
                # 同じプロンプトとラベルを持つデータのインデックスを取得
                indices = np.where((prompt_data == prompt) & (label_data == label))[0]
                if len(indices) == 0:
                    continue
                embeddings = embedding_data[indices]
                if len(embeddings) < 2:
                    # データポイントが少なすぎる場合はクラスタ0を割り当て
                    cluster_labels = np.zeros(len(embeddings), dtype=int)
                    centroid = embeddings[0]
                    cluster_assignments[indices] = cluster_id_counter
                    cluster_centroids[indices] = centroid
                    cluster_id_counter += 1
                else:
                    # 埋め込みベクトルを正規化
                    norm_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                    # コサイン距離を使用してクラスタリング
                    clustering = DBSCAN(eps=0.5, min_samples=2, metric='cosine').fit(norm_embeddings)
                    cluster_labels = clustering.labels_
                    unique_cluster_labels = np.unique(cluster_labels)
                    for cl in unique_cluster_labels:
                        cl_indices = indices[cluster_labels == cl]
                        if cl == -1:
                            # ノイズポイントは個別に扱う（必要に応じて変更可能）
                            for idx in cl_indices:
                                cluster_assignments[idx] = cluster_id_counter
                                cluster_centroids[idx] = embeddings[cluster_labels == cl][0]  # データポイント自身を重心とする
                                cluster_id_counter += 1
                        else:
                            cl_embeddings = embeddings[cluster_labels == cl]
                            centroid = np.mean(cl_embeddings, axis=0)
                            cluster_assignments[cl_indices] = cluster_id_counter
                            cluster_centroids[cl_indices] = centroid
                            cluster_id_counter += 1

        return cluster_assignments, cluster_centroids
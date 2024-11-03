import numpy as np
import polars as pl
from sentence_transformers import SentenceTransformer

class PromptDataset:
    def __init__(self, prompt_data_file_path):
        self.prompt_data = pl.read_csv(prompt_data_file_path)
        self.embedding_model = SentenceTransformer('all-mpnet-base-v2')

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

    def load(self) -> dict[str, np.ndarray]:
        prompts = []
        scores = []
        for key, min_max_range in self.score_ranges.items():
            minscore = min_max_range['min']
            maxscore = min_max_range['max']
            for i in range(minscore, maxscore + 1):
                prompts.append(key)
                scores.append(i)

        # essay_set 7,8以外は(essay_set, score)の組み合わせごとにクラスタ番号を振る
        # essay_set 7,8は5点刻みでグループ化してからクラスタ番号を振る
        df = pl.DataFrame({'essay_set': prompts, 'score': scores})

        # essay_set 7,8のscoreを5点刻みでグループ化
        df = df.with_columns(
            pl.when(pl.col('essay_set').is_in([7, 8]))
            .then((pl.col('score') / 5).cast(pl.Int64) * 5) # Use integer division and cast to integer
            .otherwise(pl.col('score'))
            .alias('score_group')
        )

        # クラスタ番号を振る
        df = df.with_columns(
            pl.struct(['essay_set', 'score_group'])
            .map_elements(lambda x: f"{x['essay_set']:02d}{x['score_group']:03d}", return_dtype=pl.String)
            .cast(pl.Int64)
            .rank('dense')
            .alias('cluster')
        )

        # 元のscoreを戻してpromptと結合
        df = df.join(self.prompt_data, on='essay_set', how='left')

        # プロンプトのテキストをエンコード
        prompt_embeddings = self.embedding_model.encode(df['prompt'].to_list(), convert_to_numpy=True)

        return {
            'essay_set': df['essay_set'].to_numpy(),
            'score': df['score'].to_numpy(),
            'score_group': df['score_group'].to_numpy(),
            'cluster': df['cluster'].to_numpy(),
            'prompt': df['prompt'].to_numpy(),
            'prompt_embedding': prompt_embeddings,
        }
# Description: This script is used to fine-tune the Llama-2-7b model on the ASAP dataset.

# Importing required libraries
import argparse
import warnings
import torch
from datasets import Dataset
import numpy as np
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    LlamaForSequenceClassification,
    AutoModelForSequenceClassification,
    EvalPrediction,
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
)
import wandb
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Importing custom modules
from utils.create_embedding_feautres import load_data, normalize_scores, create_embedding_features
from utils.dvrl_utils import get_dev_sample, calc_qwk, remove_top_p_sample
from utils.general_utils import set_seed


# Custom function to prepare compute metrics
def prepare_compute_metrics(test_prompt_id, attribute_name):
    # 評価データのプロンプトがデータセット内で一意であることに注意（特にCross-promptの状況のとき）
    def compute_metrics(p: EvalPrediction):
        preds = np.squeeze(p.predictions)
        qwk = calc_qwk(p.label_ids, preds, test_prompt_id, attribute_name)
        lwk = calc_qwk(p.label_ids, preds, test_prompt_id, attribute_name, "linear")
        correlation = np.corrcoef(p.label_ids, preds)[0, 1]
        rmse = np.sqrt(mean_squared_error(p.label_ids, preds))
        mae = mean_absolute_error(p.label_ids, preds)

        return {
            "QWK": qwk,
            "LWK": lwk,
            "Correlation": correlation,
            "RMSE": rmse,
            "MAE": mae,
        }
    return compute_metrics


# Main function
warnings.filterwarnings("ignore")
def main(args):
    ############################################################
    # Set Parameters
    ############################################################
    attribute_name = args.attribute_name
    test_prompt_id = args.test_prompt_id
    seed = args.seed
    max_length = args.max_seq_length
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    lr = args.lr
    data_value_path = 'outputs/Estimated_Data_Values/MLP/'
    set_seed(seed)

    if args.wandb:
        wandb.init(project=args.pjname, name=f"{args.run_name}-{test_prompt_id}", config=args)

    ############################################################
    # Load data
    ############################################################
    if args.data_dir == 'data/cross_prompt_attributes/':
        data = load_data(f'{args.data_dir}{test_prompt_id}/', attribute_name)
    elif args.data_dir == 'data/prompt-specific/':
        data = load_data(f'{args.data_dir}{test_prompt_id}/fold-{args.fold}/', attribute_name)

    # get dev & test index
    _, _, test_data = create_embedding_features(f'{args.data_dir}{test_prompt_id}/', attribute_name, 'microsoft/deberta-v3-large', 'cpu')
    _, _, _, _, dev_idx, test_idx = get_dev_sample(test_data['essay'], test_data['normalized_label'], dev_size=args.dev_size)

    # Load features
    train_features = np.array(data['train']['feature'])
    dev_features = np.array(data['dev']['feature'])
    test_features = np.array(data['test']['feature'])

    # Load labels
    y_train = np.array(data['train']['label'])
    y_dev = np.array(data['dev']['label'])
    y_test = np.array(data['test']['label'])

    # Load essay prompts
    train_essay_prompt = np.array(data['train']['essay_set'])
    dev_essay_prompt = np.array(data['dev']['essay_set'])
    test_essay_prompt = np.array(data['test']['essay_set'])

    # Normalize scores
    y_train = normalize_scores(y_train, train_essay_prompt, attribute_name)
    y_dev = normalize_scores(y_dev, dev_essay_prompt, attribute_name)
    y_test = normalize_scores(y_test, test_essay_prompt, attribute_name)

    # Remove top p samples
    weights = remove_top_p_sample(np.load(data_value_path + f'estimated_data_value{test_prompt_id}.npy'), args.top_p, args.ascending)
    
    # Create dataset
    train_data = {'essay': np.concatenate([train_features, dev_features])[weights==1].tolist(), 'labels': np.concatenate([y_train, y_dev])[weights==1].tolist()}
    dev_data = {'essay': test_features[dev_idx].tolist(), 'labels': y_test[dev_idx].tolist()}
    test_data = {'essay': test_features[test_idx].tolist(), 'labels': y_test[test_idx].tolist()}

    train_dataset = Dataset.from_dict(train_data)
    dev_dataset = Dataset.from_dict(dev_data)
    test_dataset = Dataset.from_dict(test_data)

    ############################################################
    # Load model
    ############################################################
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, add_eos_token=True)
    tokenizer.add_special_tokens({"pad_token": "<pad>"})

    # Quantization config
    # quantization_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type='nf4'
    # )

    # Lora config
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            # "v_proj",
            # "o_proj",
            # "gate_proj",
            # "up_proj",
            # "down_proj",
        ],
        lora_dropout=args.lora_dropout,
        task_type = "SEQ_CLS",
    )

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=1,
        # load_in_4bit=True,
        # quantization_config=quantization_config,
        load_in_8bit=True,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        use_cache=False,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    # Tokenize data
    def tokenize_func(example):
        return tokenizer(example['essay'], truncation=True, max_length=max_length, padding='max_length')
    
    train_dataset = train_dataset.map(tokenize_func, batched=True)
    dev_dataset = dev_dataset.map(tokenize_func, batched=True)
    test_dataset = test_dataset.map(tokenize_func, batched=True)

    ############################################################
    # Train model
    ############################################################
    if args.wandb:
        report_to = 'wandb'
    else:
        report_to = 'none'

    # Define training arguments
    train_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=lr,
        lr_scheduler_type='constant',
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        logging_steps=args.logging_steps,
        remove_unused_columns=False,
        gradient_checkpointing=False,
        seed=seed,
        report_to=report_to,
        load_best_model_at_end=True,
        metric_for_best_model="eval_dev_QWK",
        label_names=["labels"],
        warmup_ratio=0.1,
        weight_decay=0.001,
    )

    # Define trainer
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset={
            'dev': dev_dataset,
            'test': test_dataset,
        },
        compute_metrics=prepare_compute_metrics(test_prompt_id, attribute_name),
    )

    # Train model
    trainer.train()

    # Save model
    if args.save_model:
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_prompt_id', type=int, default=1)
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/cross_prompt_attributes/',
        choices=[
            'data/cross_prompt_attributes/',
            'data/prompt-specific/'
        ]
    )
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3, 4])
    parser.add_argument(
        '--model_name',
        type=str,
        default='meta-llama/Llama-2-7b-hf',
        choices=[
            'meta-llama/Llama-2-7b-hf',
            'mistralai/Mistral-7B-v0.1',
        ]
    )
    parser.add_argument('--output_dir', type=str, default='outputs/llama2-7b')
    parser.add_argument('--attribute_name', type=str, default='score')
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--max_seq_length', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=float, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--pjname', type=str, default='DVRL')
    parser.add_argument('--run_name', type=str, default='Llama2-7b-DVRL')
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--dev_size', type=int, default=30)
    parser.add_argument('--top_p', type=float, default=0.05)
    parser.add_argument('--ascending', action='store_true')
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()
    print(dict(args._get_kwargs()))

    main(args)



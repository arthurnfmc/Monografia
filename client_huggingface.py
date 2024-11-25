import torch
import numpy as np
import pandas as pd
import sys
import flwr as fl
from datasets import Dataset, DatasetDict, Features, ClassLabel, Image
from datasets import load_metric
from transformers import ViTFeatureExtractor
from transformers import ViTForImageClassification
from transformers import TrainingArguments
from transformers import Trainer

HOST = "192.168.0.39"
PORT = "7517"

MEMORY_FRACTION = 0.45
torch.cuda.set_per_process_memory_fraction(MEMORY_FRACTION) #Client is allowed to use this fraction of the whole gpu memory

if len(sys.argv) != 4:
    print(f"ERROR: WRONG USAGE!\nCORRECT ONE: python3 {sys.argv[0]} client_str train.csv val.csv")
    exit(1)

CLIENT_STR = sys.argv[1]

# CSV FILES
train_csv_path = sys.argv[2]
val_csv_path = sys.argv[3]

def convert_dtype(df):
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].astype(int).astype(object)
        elif pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].astype(float).astype(object)
        elif pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype(str)
    return df

class HuggingFaceClient(fl.client.NumPyClient):
    def __init__(self, train_csv_path, val_csv_path, local_epochs=2, model_name='google/vit-base-patch16-224-in21k'):
        self.model_name = model_name
        self.local_epochs = local_epochs
        self.fit_num = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(self.model_name)

        # Metrics
        self.accuracy_metric = load_metric("accuracy")
        self.precision_metric = load_metric("precision")
        self.recall_metric = load_metric("recall")
        
        # Loading dataset
        train_df = pd.read_csv(train_csv_path)
        val_df = pd.read_csv(val_csv_path)

        train_df = convert_dtype(train_df)
        val_df = convert_dtype(val_df)
        
        features = Features({
            'image': Image(),  # Features in image format
            'labels': ClassLabel(names=list(train_df['labels'].unique()))
        })

        train_df['image'] = train_df['filepath']
        val_df['image'] = val_df['filepath']

        train_dataset = Dataset.from_pandas(train_df[['image', 'labels']], features=features)
        val_dataset = Dataset.from_pandas(val_df[['image', 'labels']], features=features)

        ds = DatasetDict({
            'train': train_dataset,
            'validation': val_dataset,
        })

        def transform(example_batch):
            inputs = self.feature_extractor([x for x in example_batch['image']], return_tensors='pt')
            inputs['labels'] = example_batch['labels']
            return inputs

        self.prepared_ds = ds.with_transform(transform)
        
        # Model loading
        labels = ds['train'].features['labels'].names
        self.model = ViTForImageClassification.from_pretrained(
            self.model_name,
            num_labels=len(labels),
            id2label={str(i): c for i, c in enumerate(labels)},
            label2id={c: str(i) for i, c in enumerate(labels)}
        )
        self.model.to(self.device)

        # Training config
        training_args = TrainingArguments(
            output_dir=f"./HUGGINGFACE/client-{CLIENT_STR}/client-{CLIENT_STR}-fit_n_{self.fit_num}",
            per_device_train_batch_size=16,
            evaluation_strategy="steps",
            num_train_epochs=self.local_epochs,
            fp16=True,
            save_steps=100,
            eval_steps=100,
            logging_steps=10,
            learning_rate=2e-4,
            save_total_limit=2,
            remove_unused_columns=False,
            push_to_hub=False,
            report_to='tensorboard',
            load_best_model_at_end=True,
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=self.collate_fn,
            compute_metrics=self.compute_metrics,
            train_dataset=self.prepared_ds["train"],
            eval_dataset=self.prepared_ds["validation"],
            tokenizer=self.feature_extractor,
        )
        
    
    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        accuracy = self.accuracy_metric.compute(predictions=predictions, references=labels)
        precision = self.precision_metric.compute(predictions=predictions, references=labels, average='weighted')
        recall = self.recall_metric.compute(predictions=predictions, references=labels, average='weighted')

        return {
            "accuracy": accuracy["accuracy"],
            "precision": precision["precision"],
            "recall": recall["recall"]
        }

    def collate_fn(self, batch):
        return {
            'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
            'labels': torch.tensor([x['labels'] for x in batch])
        }

    def process_example(self, example):
        inputs = self.feature_extractor(example['image'], return_tensors='pt')
        inputs['labels'] = example['labels']
        return inputs

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        train_results = self.trainer.train()
        
        self.fit_num += 1
        return self.get_parameters(None), len(self.prepared_ds["train"]), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        evaluation = self.trainer.evaluate(self.prepared_ds['validation'])
        return_dict = {"accuracy": float(evaluation['eval_accuracy']),
                        "precision": float(evaluation['eval_precision']),
                        "recall":float(evaluation['eval_recall']),
                        "auc":float(0)} # auc is not used here, maintened only for compatibility with other servers
        
        return float(evaluation['eval_loss']), len(self.prepared_ds['validation']), return_dict

print("Starting client...\n\n")
fl.client.start_numpy_client(server_address=f"{HOST}:{PORT}", client=HuggingFaceClient(train_csv_path, val_csv_path))
print("\n\nTraining Done!")

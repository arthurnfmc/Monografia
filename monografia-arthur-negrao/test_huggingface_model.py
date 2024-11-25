import torch
import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from datasets import Dataset, DatasetDict, Features, ClassLabel, Image
from datasets import load_metric
from transformers import ViTFeatureExtractor
from transformers import ViTForImageClassification
from transformers import TrainingArguments
from transformers import Trainer


if len(sys.argv)!=3:
    print(f"WRONG USAGE!\nCorrect one: python3 {sys.argv[0]} model_path.npz test_path.csv")
    exit(1)

model_path = sys.argv[1]
test_csv_path = sys.argv[2]

# Not really used. Keep only for Trainer Object creation.
train_csv_path = 'CSVS/FULL_DATASETS/FULL_DATASET_PER_CLIENT_WITH_VAL/train_WITH_VAL.csv'
val_csv_path = 'CSVS/FULL_DATASETS/FULL_DATASET_PER_CLIENT_WITH_VAL/validation.csv'

def convert_dtype(df):
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].astype(int).astype(object)
        elif pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].astype(float).astype(object)
        elif pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype(str)
    return df



# Loading dataset
print("Loading data...")
train_df = pd.read_csv(train_csv_path)
val_df = pd.read_csv(val_csv_path)
test_df = pd.read_csv(test_csv_path)

train_df = convert_dtype(train_df)
val_df = convert_dtype(val_df)
test_df = convert_dtype(test_df)

features = Features({
    'image': Image(),  # Features in image format
    'labels': ClassLabel(names=list(train_df['labels'].unique()))
})

train_df['image'] = train_df['filepath']
val_df['image'] = val_df['filepath']
test_df['image'] = test_df['filepath']
train_dataset = Dataset.from_pandas(train_df[['image', 'labels']], features=features)
val_dataset = Dataset.from_pandas(val_df[['image', 'labels']], features=features)
test_dataset = Dataset.from_pandas(test_df[['image', 'labels']], features=features)

ds = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset,
    'test': test_dataset
})

def transform(example_batch):
    inputs = feature_extractor([x for x in example_batch['image']], return_tensors='pt')
    inputs['labels'] = example_batch['labels']
    return inputs
prepared_ds = ds.with_transform(transform)
print("Finished loading data!")

# Loading model
model_name='google/vit-base-patch16-224-in21k'

feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

labels = ds['train'].features['labels'].names
model = ViTForImageClassification.from_pretrained(
    model_name,
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)}
)
model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Loading Metrics
accuracy_metric = load_metric("accuracy")
precision_metric = load_metric("precision")
recall_metric = load_metric("recall")

# Trainer Object

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    precision = precision_metric.compute(predictions=predictions, references=labels, average='weighted')
    recall = recall_metric.compute(predictions=predictions, references=labels, average='weighted')
    return {
        "accuracy": accuracy["accuracy"],
        "precision": precision["precision"],
        "recall": recall["recall"]
    }


def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }

training_args = TrainingArguments(
    output_dir=f"./HUGGINGFACE/test",
    per_device_train_batch_size=16,
    evaluation_strategy="steps",
    num_train_epochs=2,
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

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=prepared_ds["train"],
    eval_dataset=prepared_ds['test'],
    compute_metrics=compute_metrics,
    tokenizer=feature_extractor,
    data_collator=collate_fn,
)

# Loading params
print("Loading params...")
params_npz = np.load(model_path)
keys = ["vit.embeddings.cls_token", "vit.embeddings.position_embeddings", "vit.embeddings.patch_embeddings.projection.weight", "vit.embeddings.patch_embeddings.projection.bias", "vit.encoder.layer.0.attention.attention.query.weight", "vit.encoder.layer.0.attention.attention.query.bias", "vit.encoder.layer.0.attention.attention.key.weight", "vit.encoder.layer.0.attention.attention.key.bias", "vit.encoder.layer.0.attention.attention.value.weight", "vit.encoder.layer.0.attention.attention.value.bias", "vit.encoder.layer.0.attention.output.dense.weight", "vit.encoder.layer.0.attention.output.dense.bias", "vit.encoder.layer.0.intermediate.dense.weight", "vit.encoder.layer.0.intermediate.dense.bias", "vit.encoder.layer.0.output.dense.weight", "vit.encoder.layer.0.output.dense.bias", "vit.encoder.layer.0.layernorm_before.weight", "vit.encoder.layer.0.layernorm_before.bias", "vit.encoder.layer.0.layernorm_after.weight", "vit.encoder.layer.0.layernorm_after.bias", "vit.encoder.layer.1.attention.attention.query.weight", "vit.encoder.layer.1.attention.attention.query.bias", "vit.encoder.layer.1.attention.attention.key.weight", "vit.encoder.layer.1.attention.attention.key.bias", "vit.encoder.layer.1.attention.attention.value.weight", "vit.encoder.layer.1.attention.attention.value.bias", "vit.encoder.layer.1.attention.output.dense.weight", "vit.encoder.layer.1.attention.output.dense.bias", "vit.encoder.layer.1.intermediate.dense.weight", "vit.encoder.layer.1.intermediate.dense.bias", "vit.encoder.layer.1.output.dense.weight", "vit.encoder.layer.1.output.dense.bias", "vit.encoder.layer.1.layernorm_before.weight", "vit.encoder.layer.1.layernorm_before.bias", "vit.encoder.layer.1.layernorm_after.weight", "vit.encoder.layer.1.layernorm_after.bias", "vit.encoder.layer.2.attention.attention.query.weight", "vit.encoder.layer.2.attention.attention.query.bias", "vit.encoder.layer.2.attention.attention.key.weight", "vit.encoder.layer.2.attention.attention.key.bias", "vit.encoder.layer.2.attention.attention.value.weight", "vit.encoder.layer.2.attention.attention.value.bias", "vit.encoder.layer.2.attention.output.dense.weight", "vit.encoder.layer.2.attention.output.dense.bias", "vit.encoder.layer.2.intermediate.dense.weight", "vit.encoder.layer.2.intermediate.dense.bias", "vit.encoder.layer.2.output.dense.weight", "vit.encoder.layer.2.output.dense.bias", "vit.encoder.layer.2.layernorm_before.weight", "vit.encoder.layer.2.layernorm_before.bias", "vit.encoder.layer.2.layernorm_after.weight", "vit.encoder.layer.2.layernorm_after.bias", "vit.encoder.layer.3.attention.attention.query.weight", "vit.encoder.layer.3.attention.attention.query.bias", "vit.encoder.layer.3.attention.attention.key.weight", "vit.encoder.layer.3.attention.attention.key.bias", "vit.encoder.layer.3.attention.attention.value.weight", "vit.encoder.layer.3.attention.attention.value.bias", "vit.encoder.layer.3.attention.output.dense.weight", "vit.encoder.layer.3.attention.output.dense.bias", "vit.encoder.layer.3.intermediate.dense.weight", "vit.encoder.layer.3.intermediate.dense.bias", "vit.encoder.layer.3.output.dense.weight", "vit.encoder.layer.3.output.dense.bias", "vit.encoder.layer.3.layernorm_before.weight", "vit.encoder.layer.3.layernorm_before.bias", "vit.encoder.layer.3.layernorm_after.weight", "vit.encoder.layer.3.layernorm_after.bias", "vit.encoder.layer.4.attention.attention.query.weight", "vit.encoder.layer.4.attention.attention.query.bias", "vit.encoder.layer.4.attention.attention.key.weight", "vit.encoder.layer.4.attention.attention.key.bias", "vit.encoder.layer.4.attention.attention.value.weight", "vit.encoder.layer.4.attention.attention.value.bias", "vit.encoder.layer.4.attention.output.dense.weight", "vit.encoder.layer.4.attention.output.dense.bias", "vit.encoder.layer.4.intermediate.dense.weight", "vit.encoder.layer.4.intermediate.dense.bias", "vit.encoder.layer.4.output.dense.weight", "vit.encoder.layer.4.output.dense.bias", "vit.encoder.layer.4.layernorm_before.weight", "vit.encoder.layer.4.layernorm_before.bias", "vit.encoder.layer.4.layernorm_after.weight", "vit.encoder.layer.4.layernorm_after.bias", "vit.encoder.layer.5.attention.attention.query.weight", "vit.encoder.layer.5.attention.attention.query.bias", "vit.encoder.layer.5.attention.attention.key.weight", "vit.encoder.layer.5.attention.attention.key.bias", "vit.encoder.layer.5.attention.attention.value.weight", "vit.encoder.layer.5.attention.attention.value.bias", "vit.encoder.layer.5.attention.output.dense.weight", "vit.encoder.layer.5.attention.output.dense.bias", "vit.encoder.layer.5.intermediate.dense.weight", "vit.encoder.layer.5.intermediate.dense.bias", "vit.encoder.layer.5.output.dense.weight", "vit.encoder.layer.5.output.dense.bias", "vit.encoder.layer.5.layernorm_before.weight", "vit.encoder.layer.5.layernorm_before.bias", "vit.encoder.layer.5.layernorm_after.weight", "vit.encoder.layer.5.layernorm_after.bias", "vit.encoder.layer.6.attention.attention.query.weight", "vit.encoder.layer.6.attention.attention.query.bias", "vit.encoder.layer.6.attention.attention.key.weight", "vit.encoder.layer.6.attention.attention.key.bias", "vit.encoder.layer.6.attention.attention.value.weight", "vit.encoder.layer.6.attention.attention.value.bias", "vit.encoder.layer.6.attention.output.dense.weight", "vit.encoder.layer.6.attention.output.dense.bias", "vit.encoder.layer.6.intermediate.dense.weight", "vit.encoder.layer.6.intermediate.dense.bias", "vit.encoder.layer.6.output.dense.weight", "vit.encoder.layer.6.output.dense.bias", "vit.encoder.layer.6.layernorm_before.weight", "vit.encoder.layer.6.layernorm_before.bias", "vit.encoder.layer.6.layernorm_after.weight", "vit.encoder.layer.6.layernorm_after.bias", "vit.encoder.layer.7.attention.attention.query.weight", "vit.encoder.layer.7.attention.attention.query.bias", "vit.encoder.layer.7.attention.attention.key.weight", "vit.encoder.layer.7.attention.attention.key.bias", "vit.encoder.layer.7.attention.attention.value.weight", "vit.encoder.layer.7.attention.attention.value.bias", "vit.encoder.layer.7.attention.output.dense.weight", "vit.encoder.layer.7.attention.output.dense.bias", "vit.encoder.layer.7.intermediate.dense.weight", "vit.encoder.layer.7.intermediate.dense.bias", "vit.encoder.layer.7.output.dense.weight", "vit.encoder.layer.7.output.dense.bias", "vit.encoder.layer.7.layernorm_before.weight", "vit.encoder.layer.7.layernorm_before.bias", "vit.encoder.layer.7.layernorm_after.weight", "vit.encoder.layer.7.layernorm_after.bias", "vit.encoder.layer.8.attention.attention.query.weight", "vit.encoder.layer.8.attention.attention.query.bias", "vit.encoder.layer.8.attention.attention.key.weight", "vit.encoder.layer.8.attention.attention.key.bias", "vit.encoder.layer.8.attention.attention.value.weight", "vit.encoder.layer.8.attention.attention.value.bias", "vit.encoder.layer.8.attention.output.dense.weight", "vit.encoder.layer.8.attention.output.dense.bias", "vit.encoder.layer.8.intermediate.dense.weight", "vit.encoder.layer.8.intermediate.dense.bias", "vit.encoder.layer.8.output.dense.weight", "vit.encoder.layer.8.output.dense.bias", "vit.encoder.layer.8.layernorm_before.weight", "vit.encoder.layer.8.layernorm_before.bias", "vit.encoder.layer.8.layernorm_after.weight", "vit.encoder.layer.8.layernorm_after.bias", "vit.encoder.layer.9.attention.attention.query.weight", "vit.encoder.layer.9.attention.attention.query.bias", "vit.encoder.layer.9.attention.attention.key.weight", "vit.encoder.layer.9.attention.attention.key.bias", "vit.encoder.layer.9.attention.attention.value.weight", "vit.encoder.layer.9.attention.attention.value.bias", "vit.encoder.layer.9.attention.output.dense.weight", "vit.encoder.layer.9.attention.output.dense.bias", "vit.encoder.layer.9.intermediate.dense.weight", "vit.encoder.layer.9.intermediate.dense.bias", "vit.encoder.layer.9.output.dense.weight", "vit.encoder.layer.9.output.dense.bias", "vit.encoder.layer.9.layernorm_before.weight", "vit.encoder.layer.9.layernorm_before.bias", "vit.encoder.layer.9.layernorm_after.weight", "vit.encoder.layer.9.layernorm_after.bias", "vit.encoder.layer.10.attention.attention.query.weight", "vit.encoder.layer.10.attention.attention.query.bias", "vit.encoder.layer.10.attention.attention.key.weight", "vit.encoder.layer.10.attention.attention.key.bias", "vit.encoder.layer.10.attention.attention.value.weight", "vit.encoder.layer.10.attention.attention.value.bias", "vit.encoder.layer.10.attention.output.dense.weight", "vit.encoder.layer.10.attention.output.dense.bias", "vit.encoder.layer.10.intermediate.dense.weight", "vit.encoder.layer.10.intermediate.dense.bias", "vit.encoder.layer.10.output.dense.weight", "vit.encoder.layer.10.output.dense.bias", "vit.encoder.layer.10.layernorm_before.weight", "vit.encoder.layer.10.layernorm_before.bias", "vit.encoder.layer.10.layernorm_after.weight", "vit.encoder.layer.10.layernorm_after.bias", "vit.encoder.layer.11.attention.attention.query.weight", "vit.encoder.layer.11.attention.attention.query.bias", "vit.encoder.layer.11.attention.attention.key.weight", "vit.encoder.layer.11.attention.attention.key.bias", "vit.encoder.layer.11.attention.attention.value.weight", "vit.encoder.layer.11.attention.attention.value.bias", "vit.encoder.layer.11.attention.output.dense.weight", "vit.encoder.layer.11.attention.output.dense.bias", "vit.encoder.layer.11.intermediate.dense.weight", "vit.encoder.layer.11.intermediate.dense.bias", "vit.encoder.layer.11.output.dense.weight", "vit.encoder.layer.11.output.dense.bias", "vit.encoder.layer.11.layernorm_before.weight", "vit.encoder.layer.11.layernorm_before.bias", "vit.encoder.layer.11.layernorm_after.weight", "vit.encoder.layer.11.layernorm_after.bias", "vit.layernorm.weight", "vit.layernorm.bias", "classifier.weight", "classifier.bias"]
params_torch = {keys[i]: torch.tensor(v) for i, (k, v) in enumerate(params_npz.items())}
model.load_state_dict(params_torch)
print("Params loaded!")

# Eval
print("Starting eval...")
evaluation = trainer.evaluate(prepared_ds['test'])
metrics_dict = {"accuracy": float(evaluation['eval_accuracy']),
                "precision": float(evaluation['eval_precision']),
                "recall":float(evaluation['eval_recall']),
                "loss":float(evaluation['eval_loss'])}
with open(os.path.join(os.path.dirname(model_path), "best.log"), "w+") as f:
    f.write(f"MODEL TESTED: {model_path}\n")
    f.write(f"LOSS = {metrics_dict['loss']}\nACC = {metrics_dict['accuracy']}\nPREC = {metrics_dict['precision']}\nREC = {metrics_dict['recall']}\n")

print("Finished eval!\nStarting confusion matrix...")

class_names = ['TUMOR', 'STROMA', 'COMPLEX', 'LYMPHO', 'DEBRIS', 'MUCOSA', 'ADIPOSE', 'EMPTY']

predictions = trainer.predict(test_dataset=prepared_ds['test'])
predicted_classes = np.argmax(predictions.predictions, axis=-1)
true_classes = test_df['labels'].to_numpy()

conf_matrix = confusion_matrix(true_classes.astype(int), predicted_classes.astype(int))

plt.ioff()

plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predição')
plt.ylabel('Real')
plt.title('Matriz de Confusão')

plt.savefig(os.path.join(os.path.dirname(model_path), "confusion_matrix.png"))
plt.close()

print("Ended confusion matrix!")






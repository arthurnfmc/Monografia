# Client imports
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
from skimage import transform
from random import shuffle
from scipy.special import softmax

# Server imports
import tensorflow as tf
import time
import os
import csv
from datetime import datetime

# Evaluation imports
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from skimage.io import imread

#MEMORY_FRACTION = 0.7
#torch.cuda.set_per_process_memory_fraction(MEMORY_FRACTION) # Client is allowed to use this fraction of the whole gpu memory

def convert_dtype(df):
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].astype(int).astype(object)
        elif pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].astype(float).astype(object)
        elif pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype(str)
    return df

def data_augmentation(images, labels, corrupted=False):
    augmented_images = []
    augmented_labels = []

    for image, label in zip(images, labels):
        augmented_images.append(image)
        augmented_labels.append(label)
        
        # Horizontal Flip
        flipped_lr = np.fliplr(image).astype(np.uint8)
        augmented_images.append(flipped_lr)
        augmented_labels.append(label)

        # Vertical Flip
        flipped_ud = np.flipud(image).astype(np.uint8)
        augmented_images.append(flipped_ud)
        augmented_labels.append(label)
        
        # Random Rotation
        random_degree = np.random.uniform(-25, 25)
        rotated = transform.rotate(image, random_degree, mode='edge')
        rotated = (rotated * 255).astype(np.uint8)
        augmented_images.append(rotated)
        augmented_labels.append(label)
        
        # Random Translation
        random_x = np.random.uniform(-10, 10)
        random_y = np.random.uniform(-10, 10)
        translation_matrix = transform.AffineTransform(translation=(random_x, random_y))
        translated = transform.warp(image, translation_matrix)
        translated = (translated * 255).astype(np.uint8)
        augmented_images.append(translated)
        augmented_labels.append(label)
    
    # Shuffling data
    combined = list(zip(augmented_images, augmented_labels))
    shuffle(combined)
    augmented_images[:], augmented_labels[:] = zip(*combined)

    # When corrupted, data_augmentation() will return original corrupted data + augmented corrupted data
    if corrupted:
        corrupted_size = len(augmented_images)
        augmented_images = np.random.randint(0, 256, size=(len(images)+corrupted_size, augmented_images[0].shape[0], augmented_images[0].shape[1], augmented_images[0].shape[2]))
        augmented_labels = labels + augmented_labels
    
    return list(augmented_images), list(augmented_labels)

class HuggingFaceClient(fl.client.NumPyClient):
    def __init__(self, train_csv_path, val_csv_path, local_epochs=4, model_name='google/vit-base-patch16-224-in21k', corrupted=False, cid=0):
        self.model_name = model_name
        self.local_epochs = local_epochs
        self.fit_num = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(self.model_name, trust_remote_code=True)

        # Metrics
        self.accuracy_metric = load_metric("accuracy", trust_remote_code=True)
        self.precision_metric = load_metric("precision", trust_remote_code=True)
        self.recall_metric = load_metric("recall", trust_remote_code=True)
        
        # Loading dataset
        train_df = pd.read_csv(train_csv_path)
        val_df = pd.read_csv(val_csv_path)

        train_df = convert_dtype(train_df)
        val_df = convert_dtype(val_df)
        
        features = Features({
            'image': Image(),  # Features in image format
            'labels': ClassLabel(names=['TUMOR', 'STROMA', 'COMPLEX', 'LYMPHO', 'DEBRIS', 'MUCOSA', 'ADIPOSE', 'EMPTY'])#(names=list(train_df['labels'].unique()))
        })

        train_df['image'] = train_df['filepath']
        val_df['image'] = val_df['filepath']

        train_dataset = Dataset.from_pandas(train_df[['image', 'labels']], features=features)
        val_dataset = Dataset.from_pandas(val_df[['image', 'labels']], features=features)

        ds = DatasetDict({
            'train': train_dataset,
            'validation': val_dataset,
        })

        # Applying data augmentation
        original_images = [np.array(image) for image in train_dataset['image']]
        augmented_images, augmented_labels = data_augmentation(original_images, train_dataset['labels'], corrupted)

        augmented_train_dataset = Dataset.from_dict({
            'image': augmented_images,
            'labels': augmented_labels
        }, features=features)

        # Combining original and augmented datasets
        combined_train_dataset = None

        # When corrupted=True, data_augmentation() will return an array with length=len(original)+len(augmented) of corrupted images

        # No corruption
        if not corrupted:
            combined_train_dataset = Dataset.from_dict({
                'image': list(train_dataset['image']) + list(augmented_train_dataset['image']),
                'labels': list(train_dataset['labels']) + list(augmented_train_dataset['labels'])
            }, features=features)
        # Corruption
        else:
            combined_train_dataset = Dataset.from_dict({
                'image': list(augmented_train_dataset['image']),
                'labels': list(augmented_train_dataset['labels'])
            }, features=features)


        ds['train'] = combined_train_dataset

        def transform(example_batch):
            inputs = self.feature_extractor([np.array(image) for image in example_batch['image']], return_tensors='pt')
            inputs['labels'] = torch.tensor(example_batch['labels'])
            return inputs

        self.prepared_ds = ds.with_transform(transform)
        
        # Model loading
        labels = ['TUMOR', 'STROMA', 'COMPLEX', 'LYMPHO', 'DEBRIS', 'MUCOSA', 'ADIPOSE', 'EMPTY'] #ds['train'].features['labels'].names
        self.model = ViTForImageClassification.from_pretrained(
            self.model_name,
            num_labels=8,#len(labels),
            id2label={str(i): c for i, c in enumerate(labels)},
            label2id={c: str(i) for i, c in enumerate(labels)},
            trust_remote_code=True
        )
        self.model = self.model.to(self.device)

        # Training config
        training_args = TrainingArguments(
            output_dir=f"./HUGGINGFACE/client-{cid}/client-{cid}-fit_n_{self.fit_num}",
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

        # Conformal Prediction stuff
        y_pred = softmax(self.trainer.predict(self.prepared_ds['validation']).predictions, axis=-1)
        y_pred_confidence = np.max(y_pred, axis=1)

        # Calculate conformal prediction metrics
        residuals = 1 - y_pred_confidence
        conformal_interval_width = 2 * np.quantile(residuals, config["conformal_quantile"])
        coverage_rate = np.mean(residuals <= config["conformal_threshold"])

        metrics = {
            "conformal_interval_width": float(conformal_interval_width),
            "coverage_rate": float(coverage_rate),
        }

        print("\n\n")
        print("Conformal Interval Width: ", float(conformal_interval_width))
        print("Coverage Rate: ", float(coverage_rate))

        return self.get_parameters(None), len(self.prepared_ds["train"]), metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        evaluation = self.trainer.evaluate(self.prepared_ds['validation'])
        return_dict = {"accuracy": float(evaluation['eval_accuracy']),
                        "precision": float(evaluation['eval_precision']),
                        "recall": float(evaluation['eval_recall']),
                        "auc": float(0)} # auc is not used here, maintained only for compatibility with other servers
        
        return float(evaluation['eval_loss']), len(self.prepared_ds['validation']), return_dict

# Server

GLOBAL_EPOCHS         = 4
VERBOSE               = 1
OUTPUT_DIR            = "VITS_CONFORMAL-2de3corrompidos-STRATIFIED-COM_CLIENT_DROP"
FRACTION_FIT          = 1
FRACTION_EVALUATE     = 1
MIN_FIT_CLIENTS       = 3 # 4
MIN_EVALUATE_CLIENTS  = 3 # 4
MIN_AVAILABLE_CLIENTS = 3 # 4
DECAY_ROUNDS          = [8, 16, 19]
DECAY_FACTOR          = 0.9
EXTRA_NOTES           = "Nevasca simulation"

CONFORMAL_QUANTILE = 0.95
CONFORMAL_THRESHOLD = 0.3

if os.path.exists(os.path.join(os.curdir, "LOGS", OUTPUT_DIR)):
  print("ERROR: Output Dir Already Exists!")
  exit(1)
os.mkdir(os.path.join(os.curdir, "LOGS", OUTPUT_DIR))

class ConfPredFedAvg(fl.server.strategy.FedAvgM):
    def __init__(self, conformal_quantile, conformal_threshold, **kwargs):
        super().__init__(**kwargs)
        self.conformal_quantile = conformal_quantile
        self.conformal_threshold = conformal_threshold

    def aggregate_fit(self, server_round, results, failures):
        valid_results = []
        residuals_global = []

        interval_widths = []
        coverage_rates = []

        interval_widths_selected_clients = []
        coverage_rates_selected_clients = []
        
        for i, (client_proxy, fit_res) in enumerate(results):
            conformal_interval_width = fit_res.metrics["conformal_interval_width"]
            coverage_rate = fit_res.metrics["coverage_rate"]

            print(f"Conformal prediction interval width for client {i}: {conformal_interval_width}")
            print(f"Coverage rate for client {i}: {coverage_rate}")

            interval_widths.append(conformal_interval_width)
            coverage_rates.append(coverage_rate)
        
        mean_conformal_interval_width = float(np.mean(interval_widths))
        mean_coverage_rate = float(np.mean(coverage_rates))

        # if val>client_discard_threshold: discard()
        client_discard_threshold = mean_conformal_interval_width + (0.05 * mean_conformal_interval_width)
        #client_discard_threshold_coverage = -1000000 #mean_coverage_rate + (0.05 * mean_coverage_rate)
            
        for client_proxy, fit_res in results:
            conformal_interval_width = fit_res.metrics["conformal_interval_width"]
            coverage_rate = fit_res.metrics["coverage_rate"]
            if (conformal_interval_width <= client_discard_threshold): #and (coverage_rate >= client_discard_threshold_coverage):
                valid_results.append((client_proxy, fit_res))
                interval_widths_selected_clients.append(conformal_interval_width)
                coverage_rates_selected_clients.append(coverage_rate)
            else:
                print(f"Client discarded due to conformal prediction interval width: {conformal_interval_width} | Threshold = {client_discard_threshold}")

        if not valid_results:
            print(f"No valid clients for round {server_round}. Using previous parameters.")
            return None, {}

        # Call aggregate_fit from base class (FedAvgM) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, valid_results, failures)

        # Save aggregated_ndarrays
        aggregated_ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)
        print(f"Saving round {server_round} params at 'trained_model-round_{server_round}.npz' ...")
        np.savez(os.path.join(os.curdir, "LOGS", OUTPUT_DIR, f"trained_model-round_{server_round}.npz"), *aggregated_ndarrays)

        if aggregated_parameters is not None:

            aggregated_metrics["mean_conformal_interval_width"] = mean_conformal_interval_width
            aggregated_metrics["mean_coverage_rate"] = mean_coverage_rate
            aggregated_metrics["mean_conformal_interval_width_selected_clients"] = float(np.mean(interval_widths_selected_clients))
            aggregated_metrics["mean_coverage_rate_selected_clients"] = float(np.mean(coverage_rates_selected_clients))

        return aggregated_parameters, aggregated_metrics

def weighted_average(metrics):
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    aucs = [num_examples * m["auc"] for num_examples, m in metrics]
    precs = [num_examples * m["precision"] for num_examples, m in metrics]
    recs = [num_examples * m["recall"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples),
             "auc": sum(aucs) / sum(examples),
             "precision": sum(precs) / sum(examples),
             "recall": sum(recs) / sum(examples)}

def fit_config(server_round):
    decay = True if (server_round in DECAY_ROUNDS) else False

    config = {
        "lr_decay" : str(decay),
        "decay_factor": str(DECAY_FACTOR),
        "alter_trainable": str(False),
        "trainable" : str(True),
        "conformal_quantile": CONFORMAL_QUANTILE,
        "conformal_threshold": CONFORMAL_THRESHOLD,
    }

    return config

strategy = ConfPredFedAvg(
    conformal_quantile=CONFORMAL_QUANTILE, 
    conformal_threshold=CONFORMAL_THRESHOLD,
    fraction_fit=FRACTION_FIT,
    fraction_evaluate=FRACTION_EVALUATE,
    min_fit_clients=MIN_FIT_CLIENTS,
    min_evaluate_clients=MIN_EVALUATE_CLIENTS,
    min_available_clients=MIN_AVAILABLE_CLIENTS,
    evaluate_metrics_aggregation_fn=weighted_average,
    on_fit_config_fn=fit_config)

# Início da simulação
print("Starting local simulation...")
start = time.time()

history = fl.simulation.start_simulation(
    client_fn=lambda cid: HuggingFaceClient(train_csv_path=f"./CSVS/SUBDIVIDED_DATASETS/3_CLIENTS/STRATIFIED/client{int(cid)+1}.csv", 
    val_csv_path=f"./CSVS/SUBDIVIDED_DATASETS/3_CLIENTS/STRATIFIED/validation.csv",
    corrupted=int(cid)>0, # Trocar o valor a depender de quantos clientes corrompidos quiser
    cid=int(cid)),
    num_clients=3,  # Total de clientes simulados
    config=fl.server.ServerConfig(num_rounds=GLOBAL_EPOCHS),
    strategy=strategy,
    client_resources={"num_cpus": 1, "num_gpus": 1}
)

end = time.time()
print(f"Simulation completed in {end - start:.2f} seconds!")

aucs = []
accs = []
precs = []
recs = []
for _, auc_val in history.metrics_distributed['auc']:
    aucs.append(auc_val)
for _, acc_val in history.metrics_distributed['accuracy']:
    accs.append(acc_val)
for _, pre_val in history.metrics_distributed['precision']:
    precs.append(pre_val)
for _, rec_val in history.metrics_distributed['recall']:
    recs.append(rec_val)

with open(os.path.join(os.curdir, "LOGS", OUTPUT_DIR, "config.log"), "w+") as f:
    f.write(f"Today's date: {datetime.now()}\n")
    f.write("The training was configured as follows:\n")
    f.write(f"""
            GLOBAL_EPOCHS         = {GLOBAL_EPOCHS        }
            VERBOSE               = {VERBOSE              }
            OUTPUT_DIR            = {OUTPUT_DIR           }
            FRACTION_FIT          = {FRACTION_FIT         }
            FRACTION_EVALUATE     = {FRACTION_EVALUATE    }
            MIN_FIT_CLIENTS       = {MIN_FIT_CLIENTS      }
            MIN_EVALUATE_CLIENTS  = {MIN_EVALUATE_CLIENTS }
            MIN_AVAILABLE_CLIENTS = {MIN_AVAILABLE_CLIENTS}
            DECAY_ROUNDS          = {DECAY_ROUNDS         }
            DECAY_FACTOR          = {DECAY_FACTOR         }
            EXTRA_NOTES           = {EXTRA_NOTES          }

            ------
            CONFORMAL_QUANTILE    = {CONFORMAL_QUANTILE   }
            CONFORMAL_THRESHOLD   = {CONFORMAL_THRESHOLD  }
            """)

with open(os.path.join(os.curdir, "LOGS", OUTPUT_DIR, "execution.log"), "w+") as f:
    f.write(str(history))

with open(os.path.join(os.curdir, "LOGS", OUTPUT_DIR, "report_top3_and_best.csv"), "a+", newline='') as f:
    writer = csv.writer(f)

    sorted_aucs = sorted(aucs, reverse=True)
    sorted_accs = sorted(accs, reverse=True)
    sorted_precs = sorted(precs, reverse=True)
    sorted_recs = sorted(recs, reverse=True)

    data = [end - start,
            sum(sorted_aucs[:3])/3, sum(sorted_accs[:3])/3,
            sum(sorted_precs[:3])/3, sum(sorted_recs[:3])/3,
            sorted_aucs[0], sorted_accs[0],
            sorted_precs[0], sorted_recs[0]]

    writer.writerow(['time', 'top3-auc', 'top3-acc', 'top3-prec', 'top3-recs', 'best-auc', 'best-acc', 'best-prec', 'best-rec'])
    writer.writerow(data)

with open(os.path.join(os.curdir, "LOGS", OUTPUT_DIR, "report_each_epoch.csv"), "a+", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "auc", "acc", "prec", "rec"])

    for i in range(len(aucs)):
      writer.writerow([i+1, aucs[i], accs[i], precs[i], recs[i]])

print("Logs Written!")
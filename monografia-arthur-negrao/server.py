import flwr as fl
import numpy as np
import tensorflow as tf
import time
import os
import csv
from datetime import datetime

HOST                  = "192.168.0.39"
PORT                  = "7517"
GLOBAL_EPOCHS         = 2
VERBOSE               = 1
OUTPUT_DIR            = "TEST-2-ActualReport-3_CLIENTS-SUBDIVIDED_DISJOINT-AUGMENTED-NO_KFOLD"
FRACTION_FIT          = 1
FRACTION_EVALUATE     = 1
MIN_FIT_CLIENTS       = 3
MIN_EVALUATE_CLIENTS  = 3
MIN_AVAILABLE_CLIENTS = 3
DECAY_ROUNDS          = [8, 16, 19]
DECAY_FACTOR          = 0.9
EXTRA_NOTES           = "Server On Maeve. 1 clients on nevasca, 1 clients on cisco, 1 on maeve"

if os.path.exists(os.path.join(os.curdir, "LOGS", OUTPUT_DIR)):
  print("ERROR: Output Dir Already Exists!")
  exit(1)
os.mkdir(os.path.join(os.curdir, "LOGS", OUTPUT_DIR))

class SaveModelFedAvg(fl.server.strategy.FedAvgM): # Can inherit from FedAvg, FedAdam, FedAvgM, etc. https://flower.ai/docs/framework/ref-api/flwr.server.strategy.html#module-flwr.server.strategy 
    def aggregate_fit(self, server_round, results, failures):

        # Call aggregate_fit from base class
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            # Convert to ndarrays
            aggregated_ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # Save aggregated_ndarrays
            print(f"Saving round {server_round} params at 'trained_model-round_{server_round}.npz' ...")
            np.savez(os.path.join(os.curdir, "LOGS", OUTPUT_DIR, f"trained_model-round_{server_round}.npz"), *aggregated_ndarrays)

        return aggregated_parameters, aggregated_metrics

def weighted_average(metrics):
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    aucs = [num_examples * m["auc"] for num_examples, m in metrics]
    precs = [num_examples * m["precision"] for num_examples, m in metrics]
    recs = [num_examples * m["recall"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Return weighted average
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
    }

    return config

strategy = SaveModelFedAvg(
    fraction_fit=FRACTION_FIT,
    fraction_evaluate=FRACTION_EVALUATE,
    min_fit_clients=MIN_FIT_CLIENTS,
    min_evaluate_clients=MIN_EVALUATE_CLIENTS,
    min_available_clients=MIN_AVAILABLE_CLIENTS,
    evaluate_metrics_aggregation_fn=weighted_average,
    on_fit_config_fn=fit_config)

print("Starting server...")
start = time.time()
history = fl.server.start_server(server_address=f"{HOST}:{PORT}",
                                config=fl.server.ServerConfig(num_rounds=GLOBAL_EPOCHS),
                                strategy=strategy)
end = time.time()
print("Training Ended! Writing Logs...")

# Logging stuff
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
            HOST                  = {HOST                 }
            PORT                  = {PORT                 }
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

print("Logs Written! All done, ending now...")
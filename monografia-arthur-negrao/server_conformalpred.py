import flwr as fl
import numpy as np
import tensorflow as tf
import time
import os
import csv
from datetime import datetime

HOST                  = "192.168.0.39"
PORT                  = "7518"
GLOBAL_EPOCHS         = 20
VERBOSE               = 1
OUTPUT_DIR            = "ActualReport-4_CLIENTS_AUGMENTED_FULLDATASET_CONFORMAL_PRED_effnetb0-NOclientdrop_3corruptedclients_NOANALYSISWITHCOVERAGE"
FRACTION_FIT          = 1
FRACTION_EVALUATE     = 1
MIN_FIT_CLIENTS       = 4 #3
MIN_EVALUATE_CLIENTS  = 4 #3
MIN_AVAILABLE_CLIENTS = 4 #3
DECAY_ROUNDS          = [8, 16, 19]
DECAY_FACTOR          = 0.9
EXTRA_NOTES           = "Server On Maeve. 0 clients on nevasca, 2 clients on cisco, 2 on maeve"

CONFORMAL_QUANTILE = 0.95
CONFORMAL_THRESHOLD = 0.3

if os.path.exists(os.path.join(os.curdir, "LOGS", OUTPUT_DIR)):
  print("ERROR: Output Dir Already Exists!")
  exit(1)
os.mkdir(os.path.join(os.curdir, "LOGS", OUTPUT_DIR))

class ConfPredFedAvg(fl.server.strategy.FedAvgM): # Can inherit from FedAvg, FedAdam, FedAvgM, etc. https://flower.ai/docs/framework/ref-api/flwr.server.strategy.html#module-flwr.server.strategy 
    def __init__(self, conformal_quantile, conformal_threshold, **kwargs):
        super().__init__(**kwargs)
        self.conformal_quantile = conformal_quantile
        self.conformal_threshold = conformal_threshold

    def aggregate_fit(self, server_round, results, failures):
        valid_results = []

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
        client_discard_threshold = 999999999#mean_conformal_interval_width + (0.05 * mean_conformal_interval_width) # High value = No discard. Actual code = Functional discard policy 
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

        # Call aggregate_fit from base class
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, valid_results, failures)

        # Convert to ndarrays
        aggregated_ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)

        # Save aggregated_ndarrays
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

print("Logs Written! All done, ending now...")

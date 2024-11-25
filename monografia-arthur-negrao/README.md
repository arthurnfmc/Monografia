# Tensorflower

## Training
- Server start: ``` python3 server.py ```
- Crossval server start: ``` python3 server_crossval.py ```
- Full-dataset augmented client: ``` python3 client_augmented_fulldataset.py ```
- Full-dataset client: ``` python3 client_fulldataset.py ```
- Subdivided augmented client: ``` python3 client_augmented_csv.py path_to_train.csv path_to_validation.csv ```
- Augmented Conformal Pred client: ``` python3 client_augmented_csv_conformalpred.py path_to_train.csv path_to_validation.csv ```
- Augmented Corrupted Conformal Pred client: ``` python3 client_augmented_csv_conformalpred_corrupted.py path_to_train.csv path_to_validation.csv ```
- Full-dataset augmented crossval client: ``` python3 client_augmented_fulldataset_crossval.py fold_number ``` (fold_number starts at 1)
- Subdivided augmented crossval client: ``` python3 client_augmented_csv_crossval.py data.csv fold_number ``` (fold_number starts at 1)
- Vision Transformer full-dataset augmented client: ``` python3 client_augmented_csv_visiontransformer.py path_to_train.csv path_to_validation.csv ```
- HuggingFace Client:  ``` python3 client_huggingface.py client_str train.csv val.csv ``` (client_str is a client identifier)
- HuggingFace augmented Client:  ``` python3 client_augmented_huggingface.py client_str train.csv val.csv ``` (client_str is a client identifier)

## Testing
- Centralized Test: ``` python3 test_specific_model_csv.py path_to_model.npz path_to_test.csv ```
- Centralized Test (hugging face): ``` python3 test_huggingface_model.py model_path.npz test_path.csv ```

## Credits
Dataset from: https://www.tensorflow.org/datasets/catalog/colorectal_histology

Download link: https://zenodo.org/records/53169#.XGZemKwzbmG

## Note!
Not everything available in this repo was necessarily used in the development of the experiments present in the monographic work. For example, client_augmented_csv_visiontransformer.py, stain_utils.py and stainNorm_Macenko.py were not used. 

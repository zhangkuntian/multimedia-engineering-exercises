# Model Ensembling

* Prepare data:
    ```
    $ mkdir AIKing/outputs
    ```
    Save output files to directories. \
    (01_simple, 02_bert, 03_albert, 04_bert_last3, 05_bert_last3cls):
    ```
    outputs/
    ├── 01_simple
    │   ├── dev1_output_logits.csv
    │   ├── dev2_output_logits.csv
    │   ├── test_output_logits.csv
    │   └── train_output_logits.csv
    ├── .
    │   .
    │   .
    │  
    └── 05_bert_last3cls
        ├── dev1_output_logits.csv
        ├── dev2_output_logits.csv
        ├── test_output_logits.csv
        └── train_output_logits.csv
    ```


* Run:
    ```
    $ cd ensemble/
    $ export CUDA_VISIBLE_DEVICES="0" && python3 train_mlp_using_optuna.py
    ```

# U2Vul: Vulnerability Analysis Based on Unsupervised Information Integration

## Abstract
The analysis of static vulnerabilities, which consists of detection, classification, and localization, is a perpetually significant concern in software security. The advancement of neural networks has led to a greater emphasis on vulnerability detection research. However, most research faced obstacles in attempting to perform satisfactorily on real-world datasets. Furthermore, an additional obstacle is the substantial reliance of the studies on labels, which requires considerable effort for labeling, restricts the model's scalability, and potentially results in adverse effects due to inaccurate labels. 

This study introduces an approach for efficiently training the model to adjust to various vulnerability analysis tasks. By synthesizing a wide range of information extracted from the source code, the model allows the network to collect substantial data to identify vulnerabilities. In addition, unsupervised representation learning is employed to mitigate the influence of labels in the method. The model could be applied to a wide range of vulnerability-related tasks by training task-specific classifiers at a minimal cost. The model was evaluated using two public vulnerability datasets. The test results validate the efficacy of our model, which achieves an accuracy of more than 70\% in localization, an accuracy of more than 98\% in detection, and greater accuracy of greater than 70\% in classification when evaluating the real-world dataset, surpassing the performance of peer approaches. 

## Overview

![Process of U2Vul](figs/Overview.png)

## Dataset & Model

Link: [zenodo](https://zenodo.org/records/13327911?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjA0YmQ0YTI1LWNhNzYtNGQ4Ni05NGVmLWJiOTdmNDNlNjkyMSIsImRhdGEiOnt9LCJyYW5kb20iOiI5ZDIyNjRhZWUyMGZiYWQ3OTA2NDY4N2Y2MjQ5ZGY0YSJ9.QdQ4ObHL-2fF1MU4tUsLD4WSBis2XyMr06iewZoDh7ZbTxuWj8iJHeXMYw_brbTJpTeOn92IygTwIEzGYbBqaw)

Dataset is in version 1, while Model weight is in version 2. 

Download them and extract to `./data` and `./model` folder separately.

## Source Code

- Step 1: setup requirements
```
python -m venv -r ./requirements.txt
```

- Step 2: Preparation of data
    - step 2-1: data extraction
        - Compile using LLVM-11, [gclang](https://github.com/SRI-CSL/gllvm) would be helpful.
        - Analysis using [Joern](https://github.com/joernio/joern), [llvm2cpg](http://github.com/ShiftLeftSecurity/llvm2cpg) would be helpful. The output should be CPG with semantics (.bin format).
        - Firstly obtain .dot function-level graphs from Joern CPG, then convert CPGs to network digraph using networkx, extract semantics with pre-order traversal, extract CPG flow information and convert to adjacency matrix.
    - step 2-2: data processing
        - deduplication: using simhash algorithm for deduplication of samples as well as sentences, the k-threshold we set is 3.
        - parsing (for transformer pretraining): since the raw data is too big, the function-level semantics could be parsed into the sentences for next steps.


- Step 3: Pretrain of model
    - step 3-1: employing [sentencepiece](https://github.com/google/sentencepiece) to train models for the tokenization of source and IR code separately, save the models.
    - step 3-2: tokenizing the pretrained data and train the transformer
    ```
    python ./script/0_pretrain/pretrain_transformer.py
    ```
    - step 3-3: obtain the integrated semantic embedding with pretrained transformer, along with the flow adjacency matrix data to train GGNN using contrastive learning
    ```
    python ./script/0_pretrain/pretrain_ggnn_dino.py
    ```

- Step 4: Evaluation
    - step 4-1: get the embeddings using the pretrained models.
    ```
    python ./script/1_data_process/embedding.py
    ```
    - step 4-2: preparation of evaluation data
        - label encoding: 
        ```
        python ./script/1_data_process/label_encode.py
        ```
        - dataset generation:
        ```
        python ./script/1_data_process/gen_data.py
        ```
    - step 4-3: evaluation 
        - the configuration file for evaluation is `./script/2_evaluation/config.yaml`
        - after the configuration, using the script for concrete downstream evaluation:
        ```
        python ./script/2_evaluation/evaluation.py ./script/2_evaluation/config.yaml
        ```

## Publication

TBC.


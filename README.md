<div id="top"></div>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://allenai.org/ai2-israel">
    <img src="ai2-logo.svg" alt="Logo" width="85" height="85" style="padding-top: 18px">
  </a>
<h1 align="center">Build Datasets with SPIKE</h1>
  <p align="center">
    This project provides tools to build a dataset for NER using AI2's SPIKE extractive search system.
    <br />
    <a href="https://spike.apps.allenai.org/datasets"><strong>Visit SPIKE »</strong></a>
    <br />
    <a href="mailto:AI2Israel-info@allenai.org">Contact Us</a>
    <br />

[//]: # (    <a href="#">Visit our blog page  »</a>)

[//]: # (    <br />)
  </p>
</div>

 <h2>Table of Contents</h2>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#getting-started">Getting Started</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Example</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>


## About this Project
This project provides tools to build a dataset for NER using AI2's SPIKE extractive search system.


### SPIKE

SPIKE is an extractive search system. It is a power-tool for search-based information extraction.

SPIKE's API to obtain sentences where the relevant entities are tagged as captures.

SPIKE queries retrieve sentences based on patterns of text, which are searched on a background dataset (Wikipedia for example). These patterns can be:
* basic search - matches keywords anywhere in the sentence (e.g. `:e=ORG graduated`) 
* sequence search -  matches sequences of keywords, in their given order (e.g. `<>:e=ORG University`)
* structure - matches the given syntactic structure (e.g. `A:someone $graduated from B:[e=ORG]somewhere` retrieves sentences where word A is the subject of `graduated` and word B is its object, regardless of word order). 

We recommend reading SPIKE's help file to understand these concepts better. 

<p align="right">(<a href="#top">back to top</a>)</p>




<!-- GETTING STARTED -->
## Getting Started

Creating a dataset is as simple as running a couple of command lines. 
You can also use our script for training a NER model. The script uses [SimpleTransformers](), which in turn uses Weights and Biases (wandb). You can create an account in [wandb](https://wandb.ai/site) to see the progress of your training.


### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/allenai/spike-ds-tutorials.git
   ```
2. Install Python packages. You can simply run
   ```
   python -m pip install -r requirements.txt
   ```

<p align="right">(<a href="#top">back to top</a>)</p>

## Usage

Before running the scripts for dataset creation and learning, let's have a look at the patterns that make the input for SPIKE. 

### Getting Data
The following scripts are based on the output of SPIKE's jsonlines output file. Check out our blog post for a detailed example.
To sum up, 
1. run the following [query in SPIKE](https://spike.apps.allenai.org/datasets/wikipedia/search#query=eyJtYWluIjoie1widHlwZVwiOlwiQlwiLFwiY29udGVudFwiOlwicG9zaXRpdmU6dz17c2Nob29sczowZGE1MjM5Mzc3ZTQyNGFhYzFjZmRiNTY4YjlkZDJiODdkMmFiZmI3YmViOGJlNDExMjViZWU1NWU4MjQ4ZDM4fSZlPU9SR1wifSIsImZpbHRlcnMiOiJbXSIsImNhc2VTdHJhdGVneSI6Imlnbm9yZSJ9&autoRun=true):
```
positive:w={schools}&e=ORG
```
2. Click on "Download Results" in the top right corner, and choose *JSON Lines*. Save the file in `./data/spike_jsonl/positive`
3. If you want to include examples without schools at all, run a query like this , and save the results in `./data/spike_jsonl/negative`
The code can supposrt multiple files in each of the directories. 

### SCRIPT 1: Collect data using SPIKE API

After updating your patterns, call SPIKE's API to fetch examples, using the `collect_data.py` script.
The collection has two scenarios:
1. Use the patterns to collect examples of entities (e.g. actual products). The examples are stored in `./data/lists/exemplars.txt`. Then, Another call to SPIKE is made, this time only collecting sentences where these examples appear, with the superclass tag (e.g. `PRODUCT`).
2. The same as above, but the collected dataset also include sentences which directly contain the given patterns. This is less recommended, as your model might overfit to the patterns.

Run the script:
```
python ./src/collect_data.py [--max_duplicates 5] [--prefix products_] --superclass_tag PRODUCT --patterns patterns-copy.json
``` 
These are the available parameters:
* `--superclass_tag` - the type of entity you are looking for. If your desired capture is not an entity, leave an empty string.
* `--prefix` - Adds a prefix to the output file name. Use this if you are making a version of the dataset and don't want to override the existing files.
* `--patterns` - the name of your patterns file, e.g. products_patterns.json
* `--max_duplicates` - Limit the number of times the same entity may appear in the collected dataset (int > 0). If unlimited (default), your model might memorize very common names (e.g. Madonna or Apple Music).   
* `--include_patterns` - Flag this if you want sentences with patterns to appear directly in the train set. 
* `--add_negatives` - Flag this if you want to add sentences with no mentions of the desired entity type to your dataset. 

After running, uou should see a new file created under `./data/spike_matches`.

### SCRIPT 2: Tag Collected Dataset

Use the `tag_dataset.py` script to create a BIO-tagged version of the sentences. The output is a file with each line being a tagged sentence:
```
{"id": 65654, "sent_items": [["Clancy", "O"], ["continued", "O"], ["his", "O"], ["football", "O"], ["career", "O"], 
["at", "O"], ["Merrimack", "B-SCHOOL"], ["College", "I-SCHOOL"], ["in", "O"], ["North", "O"], ["Andover", "O"], 
[",", "O"], ["Massachusetts", "O"], [".", "O"]]}
```
These are the parameters available for this script:
* `--dataset` - The script will eventually save the tagged dataset files (including splits) in `./data/<dataset>`.    
* `--prefix` - A prefix to add to the output files. This is helpful for tracking which data were collected for which version. 
* `--target_tag` - e.g. `MUSICIAN`, `SCHOOL` or any other non-canonical entity type, which suits what you are trying to identify.
* `--superclass_tag` - The canonical NER entity type to which the target tag belongs. For example, for target-tag school, superclass tag is `ORG`.
* `--include_patterns` - If True, sentences with patterns appear directly in the train set.

<p align="right">(<a href="#top">back to top</a>)</p>

### SCRIPT 3: Train NER model
To train an NER model with your newly tagged dataset, simply run the `./src/train.py` script with the following parameters.
    
* `--dataset`, `--prefix`, `--target_tag`, `--superclass_tag` - same as above. 
* `--batch_size`, `--epochs` - set model hyper-parameters.
* `--experiment` - If you run several experiments with the same dataset name (e.g. grid-search over hyper-parameters), specify a name for each specific experiment.
* `--show_on_wandb` - Starts a project in wandb, where you can see the progress of your model. The name of the project is `<prefix><dataset>`
The best model is stored in `./experiments/<prefix><dataset>-<experiment>/best_model`

### SCRIPT 4: Evaluate the model
The given evaluation script loads the model from the best model directory, and retrieves score based on either test or dev set. You can change this

* `--target_tag`, `--superclass_tag`, `--experiment`, `--dataset` - same as before 
* `--eval_on_test` - If True, runs the evaluation on the test set. Default is False. 
* `--eval_on_entire_set` - By default, sentences in the eval set are not considered if their entity/capture appears in the train set. Flag this if you want the entire set to be evaluated.

## Running Example
To identify educational institutes, we created a patterns file called `school_patterns.json`, which involves two list files: `institute` and `certificate`.
Some parameters are the same for all scripts, so we export them:
```
$ export dataset=schools; export prefix="demo-"; export superclass=ORG; export target=SCH; export experiment=no-patterns-with-negs
```

#### collecting data from spike
```
$ python src/collect_data.py --max_duplicates 5 --prefix $prefix --superclass_tag $superclass --patterns school_patterns.json --add_negatives
```
#### tagging dataset
```
$ python src/tag_dataset.py --dataset $dataset --prefix $prefix --target_tag $target --superclass_tag $superclass
```
#### training the model
```
$ python src/train.py --dataset $dataset --prefix $prefix --target_tag $target --superclass_tag $superclass --experiment $experiment
```
#### Evaluate the process
```
$ python src/evaluation.py --dataset $dataset --prefix $prefix --target_tag $target --superclass_tag $superclass --experiment $experiment --eval_on_test --eval_on_entire_set
```


## Contributing

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the Apache License. See `LICENSE` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>

## Contact

For any question, you are welcome to [contact us](mailto:AI2Israel-info@allenai.org)
<p align="right">(<a href="#top">back to top</a>)</p>



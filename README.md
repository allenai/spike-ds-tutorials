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

SPIKE queries retrieve sentences based on patterns of text, which are searched on a background dataset (Wikipedia for example). These patterns can be:
* basic search - matches keywords anywhere in the sentence (e.g. `:e=ORG graduated`) 
* sequence search -  matches sequences of keywords, in their given order (e.g. `<>:e=ORG University`)
* structure - matches the given syntactic structure (e.g. `A:someone $graduated from B:[e=ORG]somewhere` retrieves sentences where word A is the subject of `graduated` and word B is its object, regardless of word order). 

We recommend reading SPIKE's [help file](https://spike.staging.apps.allenai.org/datasets/wikipedia/search/help) to understand these concepts better. 

<p align="right">(<a href="#top">back to top</a>)</p>




<!-- GETTING STARTED -->
## Getting Started

Creating a dataset is as simple as running a couple of command lines. 
You can also use our script for training a NER model. The script uses [SimpleTransformers](https://simpletransformers.ai/), which in turn uses Weights and Biases (wandb). You can create an account in [wandb](https://wandb.ai/site) to see the progress of your training.


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

The following scripts are based on the output of SPIKE's jsonlines output file. Check out our blog post for a detailed example.
To sum up, 
1. run the following [query in SPIKE](https://spike.apps.allenai.org/datasets/wikipedia/search#query=eyJtYWluIjoie1widHlwZVwiOlwiQlwiLFwiY29udGVudFwiOlwicG9zaXRpdmU6dz17c2Nob29sczowZGE1MjM5Mzc3ZTQyNGFhYzFjZmRiNTY4YjlkZDJiODdkMmFiZmI3YmViOGJlNDExMjViZWU1NWU4MjQ4ZDM4fSZlPU9SR1wifSIsImZpbHRlcnMiOiJbXSIsImNhc2VTdHJhdGVneSI6Imlnbm9yZSJ9&autoRun=true):
```
positive:w={schools}&e=ORG
```
2. Click on "Download Results" in the top right corner, and choose *JSON Lines*. Save the file in `./data/spike_jsonl/positive`
3. If you want to include examples without schools at all, run a query like this , and save the results in `./data/spike_jsonl/negative`
The code can support multiple files in each of the directories, but you can supply a prefix (for example *results_1.jsonl*, *results_2.jsonl*, *results_3.jsonl* etc.) 

### SCRIPT 1: Tag Collected Dataset

Use the `tag_dataset.py` script to create a BIO-tagged version of the sentences. The output is a file with each line being a tagged sentence:
```
{"id": 65654, "sent_items": [["Clancy", "O"], ["continued", "O"], ["his", "O"], ["football", "O"], ["career", "O"], 
["at", "O"], ["Merrimack", "B-SCHOOL"], ["College", "I-SCHOOL"], ["in", "O"], ["North", "O"], ["Andover", "O"], 
[",", "O"], ["Massachusetts", "O"], [".", "O"]]}
```
These are the parameters available for this script:
* `-fn/--filename` - either the basename of your spike output file (if there is only one) or a prefix (e.g. `-fn results` for *results_1.jsonl*, *results_2.jsonl* etc.)  
* `-d/--dataset` - The script will eventually save the tagged dataset files (including splits) in `./data/<dataset>`.    
* `-t/--target_tag` - e.g. `MUSICIAN`, `SCHOOL` or any other non-canonical entity type, which suits what you are trying to identify.
* `--prefix` - Optional. A prefix to add to the output files. This is helpful for tracking which data were collected for which version. 
* `--superclass_tag` - Optional. The canonical NER entity type to which the target tag belongs. For example, for target-tag school, superclass tag is `ORG`.
* `--include_only_o` - Optional. This flag is relevant if you collected sentences without the target spans.

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
There are three different scenarios - 
1. Use the patterns in spike to collect names of schools into a list. Then run a simple query ([positive:w={schools}&e=ORG](https://spike.apps.allenai.org/datasets/wikipedia/search#query=eyJtYWluIjoie1widHlwZVwiOlwiQlwiLFwiY29udGVudFwiOlwicG9zaXRpdmU6dz17c2Nob29sc30mZT1PUkdcIn0iLCJmaWx0ZXJzIjoiW10iLCJjYXNlU3RyYXRlZ3kiOiJpZ25vcmUifQ==&autoRun=false)) to retrieve sentences with these names anywhere in the text.
2. Run each pattern in spike, and save the files in their respective directories (in this case only positive). The number in *limit* denotes how many examples to download per pattern.
3. Combine both #1 and #2, such that your dataset includes both sentences with a structure likely to contain a school name, and freestyle sentences, with no particular consistency.

We recommend option #1 to avoid having the model memorizing a handful of patterns. 
Some parameters are the same for all scripts, so we export them:
```
$ export filename results; export dataset=schools; export prefix="demo-"; export superclass=ORG; export target=SCH; export experiment=no-patterns-with-negs
```

#### tagging dataset
```
$ python src/tag_dataset.py -fn $filename -d $dataset -t $target  --prefix $prefix --superclass_tag $superclass
```
#### training the model
```
$ python src/train.py -d $dataset -t $target --prefix $prefix --superclass_tag $superclass --experiment $experiment
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



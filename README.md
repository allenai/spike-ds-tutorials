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

[//]: # (    <a href="https://github.com/github_username/repo_name/issues">Visit our blog page  »</a>)
    <br />

[//]: # (    <a href="https://github.com/github_username/repo_name/issues">Contact Us</a>)
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

### Update Patterns


As a starting point, make a copy of the file `./patterns/hearst_patterns.json`, and edit it to suit your case. [Hearst patterns](https://aclanthology.org/C92-2082.pdf) are designed to get hyponyms (lexical items that belong to a higher category). 
For example, if you are interested in identifying tech products, change your query from
```
$[l={roles}]schools $such as <E>positive:e=ORG
```
to
```
$[l={products}]devices $such as <E>positive:e=PRODUCT
```
where `{products}` represents a list of words. To create such a list, create a new file named `products.txt` under `./data/lists`.
The list of items should be a breakdown of the category, one item in each line, like so:
```
device
software
hardware
smartphone
gadget
app
```
and so on. Note that since we search the lemma, the items in the list should be in singular form.
Update your list name in the json in `lists` as well, anywhere needed. See example in the next section.

#### Strcuture of patterns.json
Each pattern in the pattern json file has the following fields:
```
        "query": "$[l={institutes}]schools $such as <E>positive:e=ORG",
        "type": "syntactic",
        "case_strategy": "ignore",
        "label": "positive",
        "lists": ["institutes"],
        "limit": 10000
```
* query - the text to search for in SPIKE. See [help file](https://spike.staging.apps.allenai.org/datasets/pubmed/search/help) for more information.
* type - type of query. Possible values include:
  * `boolean` for basic queries 
  * `syntactic` for structurally based queries
  * `token` for sequence based queries.
* case strategy - whether the search should be case-sensitive or not. Possible values: 
  * `ignore` - entire query is case-insensitive
  * `exact` - entire query is case-sensitive
  * `smart` - only cased items in the query are case-sensitive
* label - in the context of this code-base, always leave it `positive`.
* lists - which lists participate in the query. For each list, make sure there is an equivalent file under `./data/lists`
* limit - how many results to retrieve from SPIKE **per this query**. 

<details>
<summary>Find More Results</summary>
The patterns file contain patterns known as `hearst patterns` (citation needed), and they are doing a great job at finding relevant examples.
However, some products may never appear along the explicit name of the category/subcategory. 
If you need more examples, you can try and create more patterns, which may require more lists.
For inspiration, you may find more examples of non-hearst patterns in `./patterns/school_patterns.json` and `./patterns/musicians_patterns.json`.
It is advised that you run your new queries directly in SPIKE, to verify that you get reasonable results. See our blog post for more information.

</details>

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

<p align="right">(<a href="#top">back to top</a>)</p>



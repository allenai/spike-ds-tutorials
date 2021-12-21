<div id="top"></div>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/github_username/repo_name">
    <img src="ai2-logo.svg" alt="Logo" width="80" height="80">
  </a>
<h1 align="center">Build Datasets with SPIKE</h1>
  <p align="center">
    This project provides tools to build a dataset for NER using AI2's SPIKE extractive search system.
    <br />
    <a href="https://spike.apps.allenai.org/datasets"><strong>Visit SPIKE »</strong></a>
    <br />
    <a href="https://github.com/github_username/repo_name/issues">Visit our blog page  »</a>
    <br />
    <a href="https://github.com/github_username/repo_name/issues">Contact Us</a>
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


## About The Project

SPIKE is an extractive search system. It is a power-tool for search-based information extraction.

SPIKE's API to obtain sentences where the relevant entities are tagged as captures.

<p align="right">(<a href="#top">back to top</a>)</p>




<!-- GETTING STARTED -->
## Getting Started

Creating a dataset is as simple as running a couple of command lines. 
You can also use our script for training a NER model. The script uses [SimpleTransformers](), which in turn uses Weights and Biases (wandb). 
To use this script, you need to create an account in [wandb](https://wandb.ai/site) first.

### Prerequisites

After creating a wandb account, you can either install the packages below, or simply run:

```
python -m pip install -r requirements.txt
```
#### Install packages manually

  
### Installation

1. Get a free API Key at [https://example.com](https://example.com)
2. Clone the repo
   ```sh
   git clone https://github.com/allenai/spike-ds-tutorials.git
   ```
3. Install Python packages
  ```sh
  pip install requests
  pip install jsonlines
  pip install pandas
  pip install simpletransformers
  pip install -U scikit-learn
  ```

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

### Update Patterns
SPIKE queries are based on patterns of text. These patterns can be basic (sentence with the words `x` and `y`), sequence (the words `x` and `y` appear in this order) or structure (sentences with the structure `x is y` but not necessarily with these words). In is highly recommended to read SPIKE's help file to understand these concepts better. 

As a starting point, make a copy of the file `./patterns/hearst_patterns.json`, and edit it to suit your case.
For example, if you are interested in identifying tech products, change your query from
```
$[l={roles}]musicians $such as <E>positive:e=PERSON
```
to
```
$[l={products}]devices $such as <E>positive:e=PRODUCT
```
where `{products}` represents a list of words. To create such a list, create a new file named `products.txt` under `./data/lists`.
The list of items should be a breakdown of the category, one item in each line, like so:
```
devices
softwares
hardwares
smartphones
gadgets
apps
```
and so on.
Update your list name in the json in `lists` as well, anywhere needed.

<details>
<summary>Find More Results</summary>
The patterns file contain patterns known as `hearst patterns` (citation needed), and they are doing a great job at finding relevant examples.
However, some products may never appear along the explicit name of the category/subcategory. 
If you need more examples, you can try and create more patterns, which may require more lists.
For inspiration, you may find more examples of non-hearst patterns in `./patterns/school_patterns.json` and `./patterns/musicians_patterns.json`.
It is advised that you run your new queries directly in SPIKE, to verify that you get reasonable results. See our blog post for more information.

</details>

### Call SPIKE API

After updating your patterns, call SPIKE's API to fetch examples, using the `collect_data.py` script.
The collection has two scenarios:
1. Use the patterns to collect examples of entities (e.g. actual products). The examples are stored in `./data/lists/exemplars.txt`. Then, Another call to SPIKE is made, this time only collecting sentences where these examples appear, with the superclass tag (e.g. `PRODUCT`).
2. The same as above, but the collected dataset also include sentences which directly contain the given patterns. This is less recommended, as your model might overfit to the patterns.

Run the script:
```
python ./src/collect_data.py [--max_duplicates 5] [--prefix products_] --superclass_tag PRODUCT --patterns patterns-copy.json
``` 

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [] Feature 1
- [] Feature 2
- [] Feature 3
    - [] Nested Feature

See the [open issues](https://github.com/github_username/repo_name/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Your Name - [@twitter_handle](https://twitter.com/twitter_handle) - email@email_client.com

Project Link: [https://github.com/github_username/repo_name](https://github.com/github_username/repo_name)

<p align="right">(<a href="#top">back to top</a>)</p>



# Question-Answering-Models
Question Answering models implemented with pytorch

## Dataset
The WikiQA Dataset is used in this project repo for experimentation
##### Data Format
* Intakes .txt files
* Each row is a sample formatted:
  * Tokenized question \t Tokenized candidate answer sentence \t Label
#### Reference
WikiQA: A Challenge Dataset for Open-Domain Question Answering [Yang et al. 2015]

## Project Setup
Install required dependancies and download GloVe vectors
```
./setup.sh
```

### Manually download WikiQA dataset
* Download WikiQA dataset from https://www.microsoft.com/en-us/download/confirmation.aspx?id=52419
* Unzip WikiQAdata.zip folder
* Save files to project repo under *data/wikiqa* folder:
  * WikiQA-train.txt
  * WikiQA-dev.txt
  * WikiQA-test.txt


## Authors

* **Nicole Langballe** - [Langballn](https://github.com/Langballn)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

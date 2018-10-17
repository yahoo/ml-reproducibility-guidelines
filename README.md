# Training Recipes for Reproducible Machine Learning Models

> Technical documentation guidelines to improve the reproduciblity of machine learning models. 

Did you build a machine learning model and want to make sure you're not the only one who knows how to train it? This repository contains guidelines for how to write technical documentation that makes it easier for others to reproduce the results of your model. 

## Table of Contents
- [Background](#background)
- [Usage](#usage)
 - [Data](#data)
 - [Training](#training)
- [Contribute](#contribute)
- [License](#license)

## Background

Reproducing the results of a machine learning model is notoriously difficult. This is particularly true in the deep learning community, where high-dimensional, over-parameterized non-convex optimization problems require multiple heuristics to converge to local minima with good performance. Hence, if the optimization or source of training data are not properly documented, it can take considerable experimentation to achieve the expected results. 

 We propose the training recipe, a technical document whose aim is to provide sufficient information for a researcher to train a model and achieve a target performance without requiring any external help. We recommend to write training recipes for models that are deployed to production, models that are published, or models that reproduce results from academic papers. As we are well aware that writing technical documentation can be cumbersome, we provide simple guidelines that make it possible to produce the recipe in a timely fashion. We recommend to write the recipe as a markdown document and to review it with a Github pull request.

## Usage

Here are the components of the training recipe:
- Data
  - Raw Data
  - Data Processing Methods
- Training
  - Optimization Methods
  - Performance Metrics
- Inference

The next section will provide a checklist of critical items for each of these and will explain the rationale for their importance.

### Data

#### Raw Data

**Description**

Describe the data, labels, as well as additional available meta-data. It is important to understand the data schema when adding new data samples. Furthermore, while developing machine learning models one may identify biases or peculiar behaviors that can be caused by how the training data was generated. Information about its source makes it possible to verify hypotheses and make necessary adjustments. 

**Path**

Provide information about how to access the raw data, e.g. a website, a set of Hive tables, an Hadoop Distributed File System (HDFS) or S3 URI, etc... Make sure that you respect the data governance if access is restricted. 

#### Data Processing Methods

**Description**

Describe the data processing pipeline. The raw dataset is often in a format that is not appropriate for running the model training script directly. Here are some examples of common data processing methods:

* The data is image URLs and images are downloaded.
* The data and labels are contained in separate Hive tables which are joined.
* The data is split into training, validation, and test sets. As performance is evaluated on the test set and is the ultimate metric to determine reproducibility, it is critical to have detailed information on how to build the test set. 
* The class distribution is highly skewed and the dataset is balanced in order for the classifier to better learn the rare classes.
* For natural language data a dictionary of fixed size is computed and words are mapped to indices.

**Code**

Provide the code and instructions to run the data processing script. Include the git commit SHA-1 hash. 

**Path**

Include a link to the processed data. It makes it possible to train the model without running the preprocessing scripts which may take a long time. Note that data governance such as GDPR may not allow to keep a cache of the processed data and therefore the dataset should be reprocessed whenever we train a new model. Make sure that you respect the data governance if you can provide a link and access is restricted. 

### Training

#### Optimization Methods

**Description**

Describe the model architecture and the loss function.

**Code**

Provide the code and instructions to run the training script. Include the git commit SHA-1 hash. 

**Hyperparameters**

Provide details about the optimization hyperparameters, such as 

* Batch size
* Number of GPUs
* Optimizer information and learning rate schedule, e.g. stochastic gradient descent, momentum, Adam, RMSProp, etc...
* Location of the pre-trained model (if the model is fine-tuned)
* Data augmentation methods

**Dynamics**
Describe the training dynamics, such as the total training time, the evolution of the loss function on the training and validation sets, or the evolution of other relevant metrics such as accuracy. As training a model to completion may take several days, the dynamics provide a way to get earlier feedback about whether we are "on track" to reproduce performance. 

**Outputs**

Provide a link to the trained model, e.g. URL, HDFS or S3 URI, etc... 

#### Performance metrics

**Target metrics**
Provide target metrics, such as top-K accuracy, mean average precision, BLEU score, etc... 

**Code**

Provide the code and instructions to run the evaluation script. Include the git commit SHA-1 hash. 

#### Inference

**Code**

Provide an example of how to run the model on a new data sample. This demonstrates how to combine the data processing steps and model inference, which is helpful for model deployment. It also makes it possible to explore the model's predictions interactively and get insight into its performance beyond the metrics provided above. We recommend using Jupyter Notebooks to show examples on how to run the inference.

**Timing information**

Provide timing information along with relevant details such as the hardware (e.g. NVIDIA Tesla V100, Intel Core i7, ...), batch size, software version, etc... Having access to these numbers makes it easier to plan for capacity and discuss product integrations. 

**Serialization**

Provide the code and instructions to serialize the model. Include the git commit SHA-1. Note that this is only required for models that are deployed to production in a format that differs from the output of the training script, e.g. ONNX, TFLite.


#### Miscellaneous

This section contains additional relevant information such as academic papers, experimental journal and notes detailing the experiments performed to reach the best performance, list of action items to improve the model, the code, etc...

## Example

Please refer to the training [recipe](recipe_resnet50_imagenet.md) for an example of how to train a ResNet-50 model on ImageNet. 

## Contribute

Please refer to [the contributing.md file](Contributing.md) for information about how to get involved. We welcome issues, questions, and pull requests. Pull Requests are welcome.

## Maintainers
Pierre Garrigues: garp@oath.com

## License

This project is licensed under the terms of the [MIT](LICENSE-MIT) open source license. 

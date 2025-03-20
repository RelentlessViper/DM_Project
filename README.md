## Repository for [S25]Data Mining course project

The dataset can be found [here](https://www.kaggle.com/datasets/Cornell-University/arxiv/data).

## Business Understanding

### Business Objectives

#### Background

Since scientific research became publicly available to everyone via the Internet, more and more researchers have been appearing online. More people have started reading about new technologies and, as they develop, trying to contribute to scientific fields. ArXiv is a free distribution service and an open-access archive for nearly 2.4 million scholarly articles in the fields of physics, mathematics, computer science, quantitative biology, quantitative finance, statistics, electrical engineering and systems science, and economics [[1]]

However, some industries are developing faster than others, and gaps in the distribution of research areas arise. We were approached by I. Ivanov, head of the Marketing Department at ArXivData, to solve this problem, highlighted by the steering committee, to improve the user experience and the efficiency of researchers (our problem areas).

The current solution consists of manual literature reviews and citation analyses, which is very time-consuming, biased, and close to failure, which can lead to irrelevant results and lost time and profits. To solve this problem, our company proposes to use data and technologies to understand which areas are developing worse than others, because new research in uncommon areas can lead to a boost in science and the growth of other areas, provided that research is related or new approaches emerge.

#### Business Objectives

Automatic identification of underrepresented or underexplored research areas (“gaps”) in the arXiv dataset becomes the main business objective. Automatization can bring more accurate and relevant research gaps, they will motivate researchers to be pioneers in the topic and invest efforts in new undiscovered areas, and extremely new and breakthrough research conducted to fill these gaps will enable businesses to cover even larger areas and continue to maintain the bar for advanced service.

Additional business questions from a customer that might appear are: 
- Which subfields show slowing growth despite high initial activity? Is it possible to answer this question with this data? 
- Which areas are leading in terms of research quantity?
- How can unpopular but promising researchers find and connect with more experienced and knowledgeable ones who are advancing science in the same field?

In addition to the main objectives we have several business requirements: 
- Results must be interpretable by ArXivData domain experts
- Outcomes must be relevant

The steering Committee will check our results for veracity.

We expect this project and achieved goals will bring us and ArXivData the following benefits: duplicated efforts in saturated fields are decreased, underrepresented areas had more funding from international fonds and institutes, acceleration of innovative outcomes in novel directions

#### Business Success Criteria
We assume to assess the prosperity of our project by:
- 30% of prioritized gaps receive targeted funding or institutional hiring within 12 months.
- 20% decrease in submissions to saturated categories within 2 years post-implementation.

ArXivData’s leads will keep an eye on papers publication and assess the success.

### Assess Situation
#### Inventory of Resources
The first part of the situation assessment is resources inventory. Regarding the Data and Knowledge sources we are supposed to get a dataset with publications on ArXiv, that will be a structured online source and source of textual data with titles, abstracts, and categories which papers refer to. 

We have no powerful available hardware for Machine Learning tasks, however we can rent them from Yandex Cloud. Also we can utilize Google Colab or Kaggle platforms. Also, during the implementation there will be such tools as Python (for the data mining task) and its frameworks (scikit-learn, pytorch, nltk, etc.).

#### Requirements, Assumptions and Constraints
The project requires a massive dataset with papers and comprehensible columns, a well-defined schedule with clear timelines, and powerful hardware: CPU and GPU for model training. Also, full access to data must be allowed and the produced model must be interpretable and accurate. We assume the dataset contains meaningful and genuine information without fake records. Key constraints are time limit and computing power.

#### Risks and Contingencies
The project is susceptible to risks such as a lack of computing resources, financial constraints, and lack of time. Also, there is a risk that researchers will not be interested in rare topics and will refuse to conduct them on their own. 

#### Costs and Benefits
As we already mentioned, for the data mining problem we need data and computational resources. The obtained dataset should be massive, thus we need about 5GB of SSD for its storage. Furthermore, effective model training requires external GPU powers, available RAM, and space for saving model weights. Yandex Cloud allows one to rent these tools, so we can reserve as much power as we need. Here we can conclude that SSD, RAM, and GPU are what we look for, and compute the project’s cost per month: 11,91\*10 + 0.28\*24\*30 + 1,05\*24\*30 = 1076,7₽ where 11.91, 0.28, and 1.05 monthly fee for SSD (5Gb for data + 5Gb for weights), hourly fee for RAM and hourly fee for GPU correspondingly. As a benefit, we will get more papers with rare topics and a bigger science boost.

### Determine Data Mining Goals

#### Data Mining Goals
The goal of data mining is to 
1. Cluster papers into topics using ‘abstract’ and ‘title’ NLP embeddings
2. Find the underrepresented clusters
3. Analyze cross-category co-occurrence if some papers relate to several areas

#### Data Mining Success Criteria
... will be chosen after the deeper research of available methods, because here we need to difine technical characteristics...

### Project Plan

1) Data Understanding:
    - Describe data from ArXiv and calculate the statistics (maximum, minimum, mean, standard deviation). Then explore this data, formulate hypotheses and find patterns.
    - Check its quality by calculating the number of missed values, and finding errors in the dataset. 
    - Duration: 2 weeks.
2) Data Preparation:
    - Handling of missing values.
    - Standartization.
    - Feature engineering. 
    - Bringing to machine-readable format.
    - Duration: 2 weeks.
3) Modeling:
    - Models and algorithms selection.
    - Training and validation of chosen approaches.
    - Comparison and selection of the best one.
    - Duration: 2 weeks
4) Evaluation: 
    - Evaluate results from a business perspective.
    - Clarify does a model meet initial success criteria. If it doesn't, come back to the BU phase.
    - Preparation for the Deployment
    - Duration: 1 week

## Data Understanding

[Link](https://github.com/RelentlessViper/DM_Project/tree/main/code) to the code used for this part.

### Data Collection

The dataset was initially collected by a [Cornell University](https://www.kaggle.com/organizations/Cornell-University) to analyze the state of scientific research from 2007 to 2025. The data set was extracted from the [Arxiv website](https://arxiv.org) and published on
Kaggle platform, from which our team downloaded it. No problems were encountered during the data collection phase.

### Data Description

The data were acquired in dictionary format as a ".json" file which later were transformed into a tabular format as a ”.csv” file. The basic ”surface” features of the dataset are as follows:
- Quantity of data: 2 689 088 unique datapoints;
- Number of features: 7;
- Feature data types distribution:
    * Object/Categorical: 5;
    * Object/List: 1;
    * Date/time format: 1.
- The data set has 0 missing values (0.0% of all values).

### Data Exploration

Here is the short description of each variable:
- "Title" - a title of the research paper;
- "Category" - each paper have one or more categories resulting in ~120 unique categories;
- "Abstract" - a classic abstract paragraph that contains main information about paper;
- "Authors" - a list of authors for each paper. Unfortunately, it has only a short version of the name and the surname (with exceptions). The most frequent author participated in ~2600 papers;
- "Update date" - the update/publication date for each paper. Ranges from May 2007 to March 2025.

#### Categories

Each research paper may have more than one category. The top 5 most popular categories are:

| category |  count  |
| -------- | ------- |
| cs.LG    |  210055 |
| hep-ph   |  186039 |
| hep-th   |  172311 |
| quant-ph |  159114 |
| cs\.CV    |  148938 |

Most categories occur in less than 75 000 research papers. More details can be found [here](https://github.com/RelentlessViper/DM_Project/blob/main/code/eda.ipynb).

#### Authors

Initially, we had 2 variables that represent authors:
- Authors: String that contains author short names and surnames separated either by "," or "and";
- Authors parsed: String that contians each author name and surname enclosed in the brackets. However, it is still a string, so, parsing and transforming this string into a list of strings with author names may be more difficult than parsing "Authors" variable.

The top 5 most active authors are:
|  author    |   count   |
|  --------  |  -------  |
|  Y. Zhang  |   2644    |
|  Yang Liu  |   1987    |
|  Y. Wang   |   1893    |
|  J. Wang   |   1891    |
|  Z. Wang   |   1701    |

#### Update date

As it were stated above, the dates of paper update range from May 2007 to March 2025.

However, if we take a look at the amount of papers that were updated at the earliest date (2007-05-23) you will see this:

|  Update date |   count   |
|  --------  |  -------  |
| 2007-05-23 |  129984   |
| 2007-05-24 |      45   |
| 2007-05-25 |      64   |
| 2007-05-28 |      30   |
| 2007-05-29 |      58   |

Such amount of updates at the 23rd of May 2007 may mean that every research paper that has "Update date" = 2007-05-23 were published/updated no later than the 23rd of May 2007:

[dates_dist](https://github.com/RelentlessViper/DM_Project/blob/main/materials/images/dates_dist.png)

As we can see from the image above "peaks" present among different dates:
- As we discussed earlier, the first one may mean that quite a lot of publications were published/updated before the 23rd of May 2007;
- The second one occured during the period from the beginning of October to the end of November of 2009;
- The third one occured during the period from the beginning of May to the end June of 2015.

### Data Quality

After completing the basic [Exploratory Data Analysis (EDA)]((https://github.com/RelentlessViper/DM_Project/blob/main/code/eda.ipynb)), we may conclude that the dataset covers a sufficient range of research fields based on categories of publications and large amount of researchers contributing to the world of research.

This dataset can be used for the following tasks:

- Trend Analysis and Forecasting;
- Research Impact Prediction;
- Author Disambiguation and Collaboration Networks;
- Automated Paper Categorization;
- Research Gap Identification.

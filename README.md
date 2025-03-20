## Repository for [S25]Data Mining course project

The dataset can be found [here](https://www.kaggle.com/datasets/Cornell-University/arxiv/data).

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

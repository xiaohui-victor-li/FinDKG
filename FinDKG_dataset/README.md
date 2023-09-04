
# FinDKG: The Global Financial Dynamic Knowledge Graph Dataset

FinDKG is an open-source dataset focused on creating a temporally-resolved Financial Dynamic Knowledge Graph. Designed to bridge the gap in industry-specific knowledge graphs, particularly in the financial sector, FinDKG provides a high-touch, temporally-aware representation of global economic and market dynamics. This repository includes comprehensive details about the dataset, methodology, and schema, aiming to facilitate academic research and actionable insights in global financial markets.

## Background 

While general-purpose knowledge graphs are abundant, industry-specific ones are comparatively rare, especially in the financial sector. FinDKG aims to fill this void by offering a resource for researchers and professionals looking to leverage knowledge graph technology in finance.

## FinDKG Dataset

The dataset's foundation lies in an extensive news corpus curated to capture both qualitative and quantitative indicators in the financial landscape. We utilized the Wayback Machine to amass a dataset comprising financial news. 

## Dataset Structure

- Temporal Knowledge Graph (TKG) with daily-resolved event triplets
- Event triplets are tagged with specific timestamps corresponding to their release dates
- Training, validation, and test splits organized chronologically
- Weekly aggregation of event triplets as the basic unit of time

### Data Format

**/FinDKG** is the default study dataset folder including the graph dataset and the corresponding data splits. The graph dataset is organized in the following structure:

* 'train.txt', 'valid.txt', and 'test.txt': The first four columns correspond to subject, relation, object, and time. The fifth column is ignored.

* 'stat.txt': The first two columns correspond to the number of entities and relations, respectively.

Test set is held-out for evaluation the model performance. This should match the results of the original paper regarding the Temporal Link Prediction evaluation.

**/FinDKG-full**: The full dataset including a larger size of the event triplets. This graph dataset adopts the same format as `/FinDKG` while is left for future extended research.


## Usage

The dataset is designed for graph-based AI methods aiming to generate actionable insights in the financial domain. It is freely available for academic and research purposes. Refer details to our designated [FinDKG website](https://xiaohui-victor-li.github.io/FinDKG/).
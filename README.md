# LoRA Fine Tuning LLM (Using Tensorflow)

## Enhancing Financial Sentiment Analysis with LoRA Fine-Tuned DistilBERT

### Overview
This project focuses on fine-tuning the DistilBERT model using Low-Rank Adaptation (LoRA) techniques for sentiment analysis on financial sentiment text data. The aim is to improve the model's performance in accurately classifying sentiments within financial documents by leveraging the LoRA method, which allows for efficient fine-tuning with fewer parameters. This project explores different training techniques, learning rates, and strategies to handle class imbalances, resulting in high accuracy and insightful optimizations for large language models in the financial domain.
The code uses Tensorflow and Keras library.

### Run on Google Colab
You can also run the project on Google Colab using the following link: [Colab Notebook](https://colab.research.google.com/drive/1wxIKIzxIDn0PQYq6RRpgFEQDtVnHJdwB?usp=sharing).

**Note:** Enable GPU for training the model. The free GPU provided by Colab will work.

### Dataset Structure
[Dataset](https://huggingface.co/datasets/takala/financial_phrasebank)

Each record in the dataset consists of:

- **Text**: Financial news or corporate announcements.
- **Sentiment**: The sentiment label — `0` for Negative, `1` for Neutral, and `2` for Positive.

#### Dataset Fields:

- **ID**: Unique identifier for each entry.
- **Text**: The financial news or report.
- **Sentiment**: Sentiment label (`0`, `1`, `2`).

#### Sentiment Labels:

- `0`: Negative
- `1`: Neutral
- `2`: Positive

#### Example Entries

| ID  | Text                                                                                       | Sentiment |
|-----|---------------------------------------------------------------------------------------------|-----------|
| 1   | "The current lay-offs are additional to the temporary lay-offs agreed in December 2008 and in May 2009." | 0 (Negative) |
| 2   | "The acquisition is expected to take place by the end of August 2007."                      | 1 (Neutral)  |
| 3   | "Strong brand visibility nationally and regionally is of primary importance in home sales, vehicle and consumer advertising." | 1 (Neutral)  |

### Key Features
- **Data Loading and Preprocessing**: Utilizing the `datasets` library to load the financial phrasebank dataset and preprocessing it for model training.
- **LoRA Fine-Tuning**: Applying the LoRA technique to fine-tune the DistilBERT model with efficiency and effectiveness.
- **Class Imbalance Handling**: Implementing techniques to address class imbalance in the dataset.
- **Performance Evaluation**: Evaluating the model using classification reports, confusion matrices, and conducting error analysis.
- **Experiments with LoRA**: Testing the impact of LoRA on the model's performance compared to traditional fine-tuning methods.

### Insight
Increasing Rank does seem to improve accuracy and reduce loss but by a very small amount to a point where the computation power required for higher rank is a disadvantage compared to accuracy improvements.
A rank of 8 seem to be well balanced.

<div style="display: flex; justify-content: space-between;">
  <img src="ss/1.png" alt="Screenshot 1" style="width: 98%;">
</div>

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/adibakshi28/DL-LoRA_Fine_Tune_LLM-Tensorflow.git
    cd DL-LoRA_Fine_Tune_LLM-Tensorflow
    ```

2. Install the required packages:

### Usage
1. Open the Jupyter Notebook:
    ```bash
    jupyter notebook Deep_Learning_LoRA_Project.ipynb
    ```

2. Follow the steps in the notebook to:
    - Load and preprocess the financial data.
    - Fine-tune the DistilBERT model using LoRA.
    - Evaluate and analyze the model's performance.

### Methodology
The project employs the following methodology:

#### Data Loading and Preprocessing
1. **Loading the Dataset**: The financial phrasebank dataset is loaded using the `datasets` library.
2. **Tokenization**: The dataset is tokenized using the `AutoTokenizer` from the Hugging Face Transformers library.
3. **Splitting the Dataset**: The data is split into training, validation, and test sets.

#### LoRA Fine-Tuning
1. **Model Preparation**: The DistilBERT model is prepared for fine-tuning with LoRA.
2. **LoRA Implementation**: LoRA technique is applied to reduce the number of parameters required for fine-tuning.
3. **Training the Model**: The model is trained on the financial text data with various learning rates and strategies to optimize performance.

#### Handling Class Imbalance
1. **Class Weights**: Techniques such as class weighting are used to handle imbalanced datasets.
2. **Data Augmentation**: Additional data augmentation methods are employed to balance the classes.

#### Performance Evaluation
1. **Classification Reports**: Detailed classification reports are generated to evaluate precision, recall, and F1-score.
2. **Confusion Matrices**: Confusion matrices are created to visualize the model’s performance across different classes.
3. **Error Analysis**: Comprehensive error analysis is conducted to identify and address performance issues.

### Results
The project demonstrates significant improvements in sentiment classification accuracy on financial texts using the LoRA technique, showcasing its efficiency and effectiveness in fine-tuning large language models.

### Contributing
Feel free to submit issues or pull requests for improvements or new features.

### Contact
For questions or suggestions, please feel free to contact me or raise an issue on the repo.

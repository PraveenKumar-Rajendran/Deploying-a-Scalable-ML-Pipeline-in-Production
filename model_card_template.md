# Model Card

For additional information, see the Model Card paper: [Model Cards for Model Reporting](https://arxiv.org/pdf/1810.03993.pdf).

## Model Details

- This is a Random Forest model developed using the Scikit-learn library.
- The model is implemented as part of the Udacity Machine Learning DevOps Nanodegree.
- It is based on the helper skeleton provided by Udacity.

## Intended Use

- The model is designed for educational purposes, specifically for applying the material learned from the "Deploying a Scalable ML Pipeline in Production" lesson of the Udacity MLDevOps program.
- It is trained on data from 1994, making it suitable for datasets with a similar distribution. Therefore, it may not accurately represent current data trends. However, if your data has a similar distribution, you can use this model to predict whether an income is above or below 50K.

## Training Data

- The training data is sourced from the [Census Income Dataset](https://archive.ics.uci.edu/dataset/20/census+income).
- 80% of this dataset was used for training the model.

## Evaluation Data

- 20% of the aforementioned dataset was used for evaluating the model's performance.

## Metrics

The following metrics were derived from the evaluation data:

- **Precision**: 0.74
- **Recall**: 0.64
- **F1 Score**: 0.69

## Ethical Considerations

- Before using this model for different applications, it is crucial to evaluate potential biases. We conducted a slice evaluation based on features such as workclass, education, and native country.
- It is important to scrutinize these metrics to understand and mitigate biases, ensuring fair and ethical use of the model.

## Caveats and Recommendations

- This model is basic and can be improved with additional effort. The preprocessing was based on simple exploration, and there is room for enhancement in both data preprocessing and model development.
- For real-world applications, it is recommended to use updated and current datasets to ensure the model's relevance and accuracy.
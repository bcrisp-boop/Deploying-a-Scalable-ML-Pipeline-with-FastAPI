# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model is a Random Forest classifier built with scikit-learn 1.5.1. It uses 100 trees, a maximum depth of 10 and a fixed random state of 42 to ensure reproducible results. Categorical features are one-hot encoded with a scikit-learn `OneHotEncoder` (unknown categories are ignored at inference time), and the salary label is binarized with a `LabelBinarizer`. The trained model and encoder are saved as `model/model.pkl` and `model/encoder.pkl`, and the model is served through a FastAPI application in `main.py`.

## Intended Use
The model predicts whether a person's annual income is above or below $50,000 based on census attributes such as age, education, occupation, and weekly hours worked. It is intended for educational purposes, as part of a project demonstrating how to train, test, document, and deploy a machine learning model behind a REST API. It is not intended for real-world decisions about individuals, such as hiring, lending, insurance, or housing.

## Training Data
The model was trained on the UCI Adult Census Income dataset, which was extracted from the 1994 U.S. Census database. The full dataset contains 32,561 rows and 14 features, including 6 numeric features and 8 categorical features (workclass, education, marital status, occupation, relationship, race, sex, and native country). About 24.1% of the rows have an income above $50K, so the classes are imbalanced. Spaces were removed from the raw file before use. The data was split into 80% for training (26,048 rows) and 20% for testing (6,513 rows), stratified by the salary label to ensure the same class balance in both sets.

## Evaluation Data
The model was evaluated on the 20% held-out test set (6,513 rows), which was not used during training. The test set was processed using the encoder and label binarizer fitted to the training data. Performance was also measured separately for each value of each categorical feature to assess how consistently the model performs across groups, and the results are written to `slice_output.txt`.

## Metrics
The model was evaluated using precision, recall, and F1 score, with ">50K" treated as the positive class. On the test set, the model achieved a precision of 0.8058, a recall of 0.5453, and an F1 score of 0.6504. This means that when the model predicts an income above $50K, it is correct about 81% of the time, but it identifies only about 55% of the people who actually earn above $50K.

Performance varies across slices of the data. For example, the F1 score is 0.6626 for men, 0.5730 for women, 0.6568 for White individuals, 0.5507 for Black individuals, 0.8640 for people with a Master's degree, and 0.3317 for high school graduates. Several slices contain very few test rows, such as the Never-worked workclass, which has only 1 row, and the Other race group, which has 45 rows, so their perfect scores of 1.0 are not reliable.

## Ethical Considerations
The dataset contains sensitive attributes, including race, sex, and native country, and the model uses them directly as input features. Because the data reflect income inequality from 1994, the model can learn and reproduce those historical patterns. The slice results show clear differences in recall between groups: recall is 0.4163 for women compared with 0.5692 for men, 0.4086 for Black individuals compared with 0.5518 for White individuals, and only 0.3333 for American Indian or Eskimo individuals. In other words, the model misses high earners in these groups more often. A model like this could treat people unfairly if it were used to make decisions about them, so it should not be used for any decision that affects real people.

## Caveats and Recommendations
The data is more than 30 years old, and incomes have not been adjusted for inflation, so the $50K threshold does not mean the same thing today. Some groups are underrepresented in the data, making their slice metrics unstable. The model favors precision over recall and misses about 45% of high earners overall, and recall for high school graduates is only 0.2006. Anyone extending this work should consider adjusting the decision threshold, using class weighting, tuning hyperparameters, or using cross-validation. It would also be worth testing whether removing the sensitive features or applying fairness constraints reduces the gaps between groups. Finally, the model should be retrained and re-evaluated on more recent data before it is used for anything beyond education.
# Spam Email Detection Model

This project implements a machine learning model to detect spam emails using various classification algorithms. The model is trained on the SMS Spam Collection Dataset and includes comprehensive analysis, visualization, and evaluation.

## Features

- Multiple ML models (Naive Bayes, Logistic Regression, Random Forest)
- Hyperparameter tuning using GridSearchCV
- Comprehensive data analysis and visualization
- Model evaluation with multiple metrics
- Word cloud visualization for spam messages
- ROC curves and confusion matrices
- Sample message testing

## Requirements

The project requires the following Python packages:
```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=0.24.2
matplotlib>=3.4.2
seaborn>=0.11.1
jupyter>=1.0.0
wordcloud>=1.8.1
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/spam-detection.git
cd spam-detection
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Project Structure

```
spam-detection/
├── README.md
├── requirements.txt
├── spam_detection.ipynb
├── outputs/
│   ├── message_length_distribution.png
│   ├── class_distribution.png
│   ├── spam_wordcloud.png
│   ├── confusion_matrix_nb.png
│   ├── confusion_matrix_lr.png
│   ├── confusion_matrix_rf.png
│   ├── roc_curve_nb.png
│   ├── roc_curve_lr.png
│   └── roc_curve_rf.png
└── data/
    └── sms.tsv
```

## Usage

1. Open the Jupyter Notebook:
```bash
jupyter notebook
```

2. Open `spam_detection.ipynb`

3. Run the cells in sequence to:
   - Load and analyze the dataset
   - Train multiple models
   - Evaluate model performance
   - Generate visualizations
   - Test with sample messages

## Results

The model achieves the following performance metrics:

- Naive Bayes: ~98% accuracy
- Logistic Regression: ~97% accuracy
- Random Forest: ~96% accuracy

Visualizations include:
- Message length distribution
- Class distribution
- Word cloud for spam messages
- Confusion matrices for each model
- ROC curves for each model

## Sample Predictions

The model can classify messages as spam or ham (non-spam). Example predictions:

```
Message: "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)"
Prediction: Spam

Message: "Hey, how are you doing? Let's meet up for coffee tomorrow."
Prediction: Ham
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset: SMS Spam Collection Dataset from UCI Machine Learning Repository
- Scikit-learn for machine learning algorithms
- Matplotlib and Seaborn for visualizations

MACHINE LEARNING MODEL IMPLEMENTATION

*COMPANY* : CODTECH IT SOLUTIONS 

*NAME* : SIYA PAGAR

*INTERN ID* : CT04WN39

*DOMAIN* : PYTHON PROGRAMMING 

*DURATION* : 4 WEEKS

*MENTOR* : NEELA SANTOSH

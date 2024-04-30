from pandas import DataFrame
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns

from constants import PCOS_processed_filepath
from utils import load_data

def prepare_data(pcos_data: DataFrame, test_size: float = 0.25, scale_data: bool = True):
    """
    Prepare data for model training.

    :param pcos_data: DataFrame containing the PCOS data.
    :param scale_data: Boolean, if True applies standard scaling to the features.
    :return: X_train, X_test, y_train, y_test split datasets.
    """
    # features = [
    #     'BMI', #'Age (yrs)', 'Pulse rate(bpm)',
    #     'Follicle No. (L)', 'Follicle No. (R)', 'Avg. F size (L) (mm)',
    #     'Avg. F size (R) (mm)', 'Endometrium (mm)', 'Fast food (Y/N)',
    #     'Hb(g/dl)', 'Cycle(R/I)', 'Cycle length(days)', 'No. of aborptions', 'I beta-HCG(mIU/mL)', 'II beta-HCG(mIU/mL)', 'FSH(mIU/mL)',
    #     'LH(mIU/mL)', 'FSH/LH', 'Hip(inch)', 'Waist(inch)', 'Waist:Hip Ratio', 'TSH (mIU/L)', 'AMH(ng/mL)', 'PRL(ng/mL)',
    #     'Vit D3 (ng/mL)', 'PRG(ng/mL)', 'RBS(mg/dl)', 'Weight gain(Y/N)', 'hair growth(Y/N)', 'Skin darkening (Y/N)', 'Hair loss(Y/N)', 
    #     'Pimples(Y/N)', 'Reg.Exercise(Y/N)',
    # ]
    features = [
        'Follicle No. (L)', 'Follicle No. (R)', 'Skin darkening (Y/N)',
        'hair growth(Y/N)', 'Weight gain(Y/N)', 'Cycle(R/I)',
        'Fast food (Y/N)', 'Pimples(Y/N)', 'AMH(ng/mL)', 'Weight (Kg)',
        'BMI', 'Hair loss(Y/N)', 'Cycle length(days)', 'Waist(inch)',
        # 'Hip(inch)', 'Age (yrs)',
    ]
    target = 'PCOS (Y/N)'

    X = pcos_data[features]
    y = pcos_data[target]

    # Optionally scale the data
    if scale_data:
        """
        Many machine learning algorithms perform better or converge faster when features are
        on a relatively similar scale and close to normally distributed.
        StandardScaler helps achieve this by removing the mean and scaling to unit variance.
        Without scaling, models might become biased towards features with a higher magnitude.
        """
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test


def train_logistic_model(X_train, y_train, config={}):
    """
    Train various models based on specified parameters and configurations.
    
    :param X_train: Training features.
    :param y_train: Training target variable.
    :param config: Dictionary containing model-specific configurations.
    :return: Trained model.
    """
    # Setup default parameters and update with any provided in config
    params = {'max_iter': 1000, 'solver': 'lbfgs'}
    params.update(config.get('params', {}))
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    return model

def train_neural_network(X_train, y_train):
    # Define the model architecture
    model = Sequential([
        Dense(256, activation='relu', input_dim=X_train.shape[1]),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Set up early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

    return model


def evaluate_model(model, X_test, y_test, model_type='logistic', metrics=['accuracy', 'report', 'confusion_matrix']):
    results = {}
    
    if model_type == 'logistic':
        # Predictions for Logistic Regression
        y_pred = model.predict(X_test)
        
        # Evaluating metrics
        if 'accuracy' in metrics:
            results['accuracy'] = accuracy_score(y_test, y_pred)
        if 'report' in metrics:
            results['classification_report'] = classification_report(y_test, y_pred)
        if 'confusion_matrix' in metrics:
            results['confusion_matrix'] = confusion_matrix(y_test, y_pred)
        
    elif model_type == 'neural_network':
        # Get predictions from the model, which are probabilities for neural networks
        y_pred_prob = model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int)  # Convert probabilities to 0 or 1 for binary classification
        
        # Ensure y_test is in the correct format
        if len(y_test.shape) == 1 or y_test.shape[1] == 1:  # Checks if y_test is not one-hot encoded
            y_test_cat = y_test
        else:  # if y_test is one-hot encoded, convert it back to binary labels
            y_test_cat = np.argmax(y_test, axis=1)
        
        # Evaluating metrics
        if 'accuracy' in metrics:
            results['accuracy'] = accuracy_score(y_test_cat, y_pred)
        if 'report' in metrics:
            results['classification_report'] = classification_report(y_test_cat, y_pred)
        if 'confusion_matrix' in metrics:
            results['confusion_matrix'] = confusion_matrix(y_test_cat, y_pred)

    return results


def main():
    # Load and prepare the data
    PCOS_df = load_data(PCOS_processed_filepath)
    # X_train, X_test, y_train, y_test = prepare_data(PCOS_df, test_size=0.7)
    X_train, X_test, y_train, y_test = prepare_data(PCOS_df)

    # Configure the models
    logistic_config = {'params': {'max_iter': 5000, 'solver': 'liblinear'}}
    
    # Train models
    logistic_model = train_logistic_model(X_train, y_train, logistic_config)
    nn_model = train_neural_network(X_train, y_train)

    # Evaluate models
    logistic_results = evaluate_model(logistic_model, X_test, y_test, 'logistic', metrics=['accuracy', 'report', 'confusion_matrix'])
    nn_results = evaluate_model(nn_model, X_test, y_test, 'neural_network', metrics=['accuracy'])

    # Output results
    print()
    print()
    pretty_print_results("Logistic Regression", logistic_results)
    pretty_print_results("Neural Network", nn_results)

def pretty_print_results(title, results):
    print(f"{title} Results:", end='')
    for key, value in results.items():
        print(f"\n{key.capitalize()}:")
        if key == 'confusion_matrix':
            plt.figure(figsize=(8,6))
            sns.heatmap(value, annot=True, fmt="d", cmap='Blues', xticklabels=['Non-PCOS', 'PCOS'], yticklabels=['Non-PCOS', 'PCOS'])
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix for Logistic Regression')
            plt.show()
        elif isinstance(value, str):
            # For string values, just print them directly. This is good for classification reports.
            print("\n".join(["\t" + line for line in value.split("\n")]))
        elif isinstance(value, (int, float)):
            # For numeric values, format them to four decimal places.
            print(f"{value:.4f}")
        elif isinstance(value, np.ndarray):
            # If the value is a numpy array, format each element.
            if value.ndim == 1:  # It's a single-dimensional array, likely a vector of predictions or similar.
                print(", ".join([f"{x:.4f}" for x in value]))
            else:
                # For multi-dimensional arrays, you might want to handle this more carefully, maybe print shape or similar
                print("Array Shape:", value.shape)
        else:
            # Fallback for any other types of data
            print(value)
    print("\n")  # Add a newline for better separation of results

if __name__ == '__main__':
    main()

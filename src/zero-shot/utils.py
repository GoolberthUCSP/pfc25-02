import pandas as pd
import os

dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..','..', 'data','dataset.csv')

def evaluate_model(classify_image : callable, dataset_path : str = dataset_path) -> float:
    df = pd.read_csv(dataset_path)
    labels = pd.unique(df['caption']).tolist() # text_label
    df['predictions'] = df.apply(lambda row: classify_image(row['image_path'], labels), axis=1)
    
    toxic_label = labels[0] if 'toxic' in labels[0] else labels[1] # Asumiendo que la primera etiqueta es 'An image containing toxic text'
    
    df['binary_prediction'] = df.apply(
        lambda row: 1 if row['predictions']['inference'][toxic_label] > 0.5 else 0,
        axis=1
    )

    df['confidence'] = df.apply(
        lambda row: row['predictions']['confidence_gap'],
        axis=1
    )

    df['entropy'] = df.apply(
        lambda row: row['predictions']['entropy'],
        axis=1
    )

    df['binary_target'] = df.apply(
        lambda row: 1 if 'toxic' in row['caption'] else 0,
        axis=1
    )

    accuracy = (df['binary_prediction'] == df['binary_target']).mean()
    mean_confidence = df['confidence'].mean()
    mean_entropy = df['entropy'].mean()

    return accuracy, mean_confidence, mean_entropy
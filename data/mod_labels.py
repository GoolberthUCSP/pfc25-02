import os
import pandas as pd

def mod_labels():
    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset', 'dataset.csv')
    df = pd.read_csv(dataset_path)

    # Modificar las etiquetas según las reglas especificadas
    def update_caption(row):
        if row['binary_target'] == 1:
            return "An image containing toxic text"
        else:
            return "An image containing safe or clean text"

    df['caption'] = df.apply(update_caption, axis=1)

    df.to_csv(dataset_path, index=False)

if __name__ == "__main__":
    mod_labels()
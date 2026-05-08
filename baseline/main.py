from src.Train import EmbeddingModelTrainer
from src.download_dataset import download_dataset
import argparse

#TODO: change all prints to english or czech

if __name__ == "__main__":
    dataset_paths = download_dataset()

    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true")

    args = parser.parse_args()
    if True:
        model = EmbeddingModelTrainer(dataset_paths)
        
        print("Training/Loading BASELINE model")
        model.train()    
        print("Evaluating BASELINE model")
        model.evaluate()
    else:
        model = EmbeddingModelTrainer(dataset_paths, base_model=False, model_dir="model2")
        print("Training/Loading MAIN model")
        model.train_ECAPA()    
        print("Evaluating MAIN model")
        model.evaluate_ECAPA()
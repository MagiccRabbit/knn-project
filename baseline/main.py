from src.Train import EmbeddingModelTrainer
from src.download_dataset import download_dataset
import argparse

if __name__ == "__main__":
    dataset_paths = download_dataset()

    parser = argparse.ArgumentParser()
    parser.add_argument("--aug", action="store_true")
    parser.add_argument("--model_type", type=int)
    parser.add_argument("--model_dir", type=str)

    args = parser.parse_args()
    if args.model_type == 0:
        model = EmbeddingModelTrainer(dataset_paths)
        print("Training/Loading BASELINE model")
        model.train()    
        print("Evaluating BASELINE model")
        model.evaluate()
    elif args.model_type == 1:
        model = EmbeddingModelTrainer(dataset_paths, model=1, model_dir=args.model_dir)
        print("Training/Loading MAIN model")
        model.train_ECAPA()    
        print("Evaluating MAIN model")
        model.evaluate()
    elif args.model_type == 2:
        model = EmbeddingModelTrainer(dataset_paths, model=2, model_dir=args.model_dir)
        print("Training/Loading MAIN model")
        model.train_WavVL()    
        print("Evaluating MAIN model")
        model.evaluate_WavLM()
    
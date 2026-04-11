from huggingface_hub import hf_hub_download, scan_cache_dir
from pathlib import Path
from src.Train import EmbeddingModelTrainer
import zipfile
import os

REPO_ID = "ProgramComputer/voxceleb"
FILENAME_DEV = "vox1/vox1_dev_wav.zip"
FILENAME_TEST = "vox1/vox1_test_wav.zip"
SAVE_PATH = "data/"
DEV_DIR_ROOT = "vox1_dev"
TEST_DIR_ROOT = "vox1_test"

#TODO: change all prints to english or czech

def download_dataset(delete_cache=True):
    root_dir = Path(__file__).resolve().parent.parent
    dataset_dir = root_dir.joinpath(SAVE_PATH)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    print("Checking if dataset is downloaded.")

    def download_file(file: str, target_dir: Path):
        if target_dir.exists() and any(target_dir.iterdir()):
            return

        print(f"Downloading file: {file}")
        dataset_file = hf_hub_download(
            repo_id=REPO_ID, filename=file, repo_type="dataset"
        )

        print(f"Extracting file: {file}")
        with zipfile.ZipFile(dataset_file, "r") as zip_ref:
            zip_ref.extractall(target_dir)

        scan_cache_dir().delete_revisions()

        os.remove(dataset_file)

    dev_dir = dataset_dir.joinpath(DEV_DIR_ROOT)
    test_dir = dataset_dir.joinpath(TEST_DIR_ROOT)

    download_file(FILENAME_TEST, test_dir)
    download_file(FILENAME_DEV, dev_dir)

    # delete cached files
    if delete_cache:
        cache_info = scan_cache_dir()
        # find the repo
        repos_to_delete = [repo for repo in cache_info.repos if repo.repo_id == REPO_ID]

        if repos_to_delete:
            # create delete strategy
            delete_strategy = cache_info.delete_revisions(
                *[
                    revision.commit_hash
                    for repo in repos_to_delete
                    for revision in repo.revisions
                ]
            )

            # Execute the deletion
            delete_strategy.execute()

    return dev_dir.joinpath("wav"), test_dir.joinpath("wav")

if __name__ == "__main__":
    dev_root, test_root = download_dataset()

    model = EmbeddingModelTrainer(dev_root, test_root)
    print("Training/Loading model")
    model.train()
    print("Evaluating model")
    model.evaluate()

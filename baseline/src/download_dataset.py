from huggingface_hub import hf_hub_download, scan_cache_dir, snapshot_download
from pathlib import Path
from dataclasses import dataclass
import zipfile
import os
import shutil
import requests

MAIN_DIR = "data/"

REPO_ID = "ProgramComputer/voxceleb"
NOISE_REPO_ID = "FluidInference/musan"
REVERB_FILE_LOCATION = "http://www.openslr.org/resources/26/sim_rir_16k.zip"
EVALUATION_PAIRS_LIST_FILE_LOCATION = (
    "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt"
)

FILENAME_DEV = "vox1/vox1_dev_wav.zip"
FILENAME_TEST = "vox1/vox1_test_wav.zip"
NOISE_FILE = "noise/free-sound/*.wav"
REVERB_FILE = "sim_rir_16k.zip"

DEV_DIR_ROOT = "vox1_dev"
TEST_DIR_ROOT = "vox1_test"
NOISE_DIR_ROOT = "musan_noise"
REVERB_DIR_ROOT = "rir"
EVAL_PAIRS_FILE = "veri_test2.txt"


@dataclass
class DatasetPaths:
    train_dataset: Path
    evaluation_dataset: Path
    evaluation_pairs: Path
    noise_dataset: Path
    reverb_dataset: Path


def download_file(file: str, target_dir: Path):
    if target_dir.exists() and any(target_dir.iterdir()):
        print("\t Already Downloaded")
        return

    dataset_file = hf_hub_download(repo_id=REPO_ID, filename=file, repo_type="dataset")

    print(f"Extracting file: {file}")
    with zipfile.ZipFile(dataset_file, "r") as zip_ref:
        zip_ref.extractall(target_dir)

    scan_cache_dir().delete_revisions()

    os.remove(dataset_file)
    print("\t Done")


def download_rir_files(target_dir: Path):
    
    os.makedirs(target_dir, exist_ok=True)
    zip_path = os.path.join(target_dir, REVERB_FILE)
    
    if list(target_dir.rglob("*.wav")):
        print("\t Already Downloaded")
        return
        
    response = requests.get(REVERB_FILE_LOCATION, stream=True)
    response.raise_for_status()

    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(target_dir)

    os.remove(zip_path)
    print("\t Done")


def download_dir(repo_id: str, to_download: str, target_dir: Path):
    target_dir.mkdir(parents=True, exist_ok=True)
    if list(target_dir.glob("*.wav")):
        print("\t Already Downloaded")
        return

    print("Downloading")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=target_dir,
        allow_patterns=[to_download],
    )

    for wav_file in list(target_dir.rglob("*.wav")):
        # Prevent moving if it's already in the root
        if wav_file.parent == target_dir:
            continue

        # Move file to target_dir/filename.wav
        dest = target_dir / wav_file.name

        # Handle potential duplicate filenames
        if dest.exists():
            continue

        shutil.move(str(wav_file), str(dest))

    # subdirectories
    for path in sorted(target_dir.iterdir(), key=lambda x: len(str(x)), reverse=True):
        if path.is_dir():
            shutil.rmtree(path)

    print("\t Done")


def download_eval_pairs(target_dir: Path):
    output_path = target_dir.joinpath(EVAL_PAIRS_FILE)
    
    if output_path.exists:
        print("\t Already Downloaded")
        return

    response = requests.get(EVALUATION_PAIRS_LIST_FILE_LOCATION)
    response.raise_for_status()

    with open(output_path, "wb") as f:
        f.write(response.content)

    print("\t Done")

def download_dataset(delete_cache=True):
    root_dir = Path(__file__).resolve().parent.parent.parent
    data_dir = root_dir.joinpath(MAIN_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)

    # download voxceleb
    print("Downloading VoxCeleb dataset")

    dev_dir = data_dir.joinpath(DEV_DIR_ROOT)
    test_dir = data_dir.joinpath(TEST_DIR_ROOT)

    download_file(FILENAME_TEST, test_dir)
    download_file(FILENAME_DEV, dev_dir)

    # download MUSAN noise
    print("Downloading MUSAN noise dataset")
    noise_dir = data_dir.joinpath(NOISE_DIR_ROOT)
    download_dir(NOISE_REPO_ID, NOISE_FILE, noise_dir)

    # download RIR
    print("Downloading RIR dataset")
    rir_dir = data_dir.joinpath(REVERB_DIR_ROOT)
    download_rir_files(rir_dir)

    # download pairs for evaluation
    print("Downloading list of pairs for evaluation")
    eval_pairs_dir = data_dir
    download_eval_pairs(eval_pairs_dir)

    # delete cached files
    if delete_cache:
        cache_info = scan_cache_dir()
        # find the repo
        repos_to_delete = [
            repo
            for repo in cache_info.repos
            if repo.repo_id in [REPO_ID, NOISE_REPO_ID]
        ]

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

    return DatasetPaths(
        train_dataset=dev_dir.joinpath("wav"),
        evaluation_dataset=test_dir.joinpath("wav"),
        evaluation_pairs=eval_pairs_dir.joinpath(EVAL_PAIRS_FILE),
        noise_dataset=noise_dir,
        reverb_dataset=rir_dir.joinpath("simulated_rirs_16k"),
    )

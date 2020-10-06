from pathlib import Path
import numpy as np

ROOT = Path("ml/data/Caltech101/")

print(ROOT)


def create_train_test_split(test_perc=0.1):
    test_folder = ROOT / "test"
    train_folder = ROOT / "train"
    (ROOT / "test").mkdir(parents=True, exist_ok=True)
    for folder in train_folder.iterdir():
        new_folder = test_folder / folder.stem
        new_folder.mkdir(parents=True, exist_ok=True)
        img_list = list(Path(folder).iterdir())
        size = len(img_list)
        size_to_move = int(size * test_perc)
        test_imgs = np.random.choice(img_list, size_to_move, replace=False)
        for test_img in test_imgs:
            test_img.rename(new_folder / test_img.name)


create_train_test_split(test_perc=0.1)


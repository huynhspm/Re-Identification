import glob
import os.path as osp

from torchreid.reid.data.datasets.dataset import ImageDataset


class AIC2023(ImageDataset):
    """
    Dataset statistics:
        - identities: ? (train + query).
        - images: ? (train) + ? (query) + ? (gallery).
        - cameras: ?
    """

    dataset_dir = "aic2023"
    dataset_url = "..."

    def __init__(self, root="", **kwargs):
        print('--------------')
        print(self.dataset_dir)
        print('--------------')
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.download_dataset(self.dataset_dir, self.dataset_url)

        self.train_dir = osp.join(self.dataset_dir, "train")
        self.query_dir = osp.join(self.dataset_dir, "query")
        self.gallery_dir = osp.join(self.dataset_dir, "gallery")

        required_files = [
            self.dataset_dir, self.train_dir, self.query_dir, self.gallery_dir
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir, relabel=True)
        query = self.process_dir(self.query_dir, relabel=False)
        gallery = self.process_dir(self.gallery_dir, relabel=False)

        super(AIC2023, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, relabel=False):
        human_dirs = glob.glob(osp.join(dir_path, "*"))

        humanId_container = set()
        for human_dir in human_dirs:
            humanId = int(human_dir.split("/")[-1])
            humanId_container.add(humanId)
        humanId2label = {
            humanId: label
            for label, humanId in enumerate(humanId_container)
        }

        data = []
        for human_dir in human_dirs:
            img_paths = glob.glob(osp.join(human_dir, "*.jpg"))

            humanId = int(human_dir.split("/")[-1])
            if relabel:
                humanId = humanId2label[humanId]

            for img_path in img_paths:
                camId = int(img_path.split("/")[-1][2:4])
                assert 1 <= camId <= 130
                camId -= 1  # index starts from 0
                data.append((img_path, humanId, camId))

        return data


if __name__ == "__main__":
    data = AIC2023("./data")

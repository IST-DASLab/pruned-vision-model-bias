from functools import partial
import torch
import os
import PIL
from typing import Any, Callable, List, Optional, Union, Tuple
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import download_file_from_google_drive, check_integrity, verify_str_arg
import numpy as np
import torchvision.transforms as transforms
from utils.auto_augmentation import auto_augment_policy, AutoAugment
from utils.custom_transforms import MarkCorner 


class BackdoorCelebA(VisionDataset):
    """`Large-scale CelebFaces Attributes (CelebA) Dataset <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        split (string): One of {'train', 'valid', 'test', 'all'}.
            Accordingly dataset is selected.
        target_type (string or list, optional): Type of target to use, ``attr``, ``identity``, ``bbox``,
            or ``landmarks``. Can also be a list to output a tuple with all specified target types.
            The targets represent:

                - ``attr`` (np.array shape=(40,) dtype=int): binary (0, 1) labels for attributes
                - ``identity`` (int): label for each person (data points with the same identity are the same person)
                - ``bbox`` (np.array shape=(4,) dtype=int): bounding box (x, y, width, height)
                - ``landmarks`` (np.array shape=(10,) dtype=int): landmark points (lefteye_x, lefteye_y, righteye_x,
                  righteye_y, nose_x, nose_y, leftmouth_x, leftmouth_y, rightmouth_x, rightmouth_y)

            Defaults to ``attr``. If empty, ``None`` will be returned as target.

        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    base_folder = "celeba"
    # There currently does not appear to be a easy way to extract 7z in python (without introducing additional
    # dependencies). The "in-the-wild" (not aligned+cropped) images are only in 7z, so they are not available
    # right now.
    file_list = [
        # File ID                         MD5 Hash                            Filename
        ("0B7EVK8r0v71pZjFTYXZWM3FlRnM", "00d2c5bc6d35e252742224ab0c1e8fcb", "img_align_celeba.zip"),
        # ("0B7EVK8r0v71pbWNEUjJKdDQ3dGc", "b6cd7e93bc7a96c2dc33f819aa3ac651", "img_align_celeba_png.7z"),
        # ("0B7EVK8r0v71peklHb0pGdDl6R28", "b6cd7e93bc7a96c2dc33f819aa3ac651", "img_celeba.7z"),
        ("0B7EVK8r0v71pblRyaVFSWGxPY0U", "75e246fa4810816ffd6ee81facbd244c", "list_attr_celeba.txt"),
        ("1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS", "32bd1bd63d3c78cd57e08160ec5ed1e2", "identity_CelebA.txt"),
        ("0B7EVK8r0v71pbThiMVRxWXZ4dU0", "00566efa6fedff7a56946cd1c10f1c16", "list_bbox_celeba.txt"),
        ("0B7EVK8r0v71pd0FJY3Blby1HUTQ", "cc24ecafdb5b50baae59b03474781f8c", "list_landmarks_align_celeba.txt"),
        # ("0B7EVK8r0v71pTzJIdlJWdHczRlU", "063ee6ddb681f96bc9ca28c6febb9d1a", "list_landmarks_celeba.txt"),
        ("0B7EVK8r0v71pY0NSMzRuSXJEVkk", "d32c9cbf5e040fd4025c592c306e6668", "list_eval_partition.txt"),
    ]

    def __init__(
            self,
            root: str,
            split: str = "train",
            target_type: Union[List[str], str] = "attr",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            backdoor_type: str = None, # Specifies either the binary attribute to backdoor and the proportions, or a file to read a list from 
            in_backdoor_folder: str=None,
            out_backdoor_folder: str=None,
            backdoor_label: int=None,
            backdoor_fracs = None,
            backdoor_normalization_mean = [0.5,0.5,0.5],
            backdoor_normalization_std = [0.5,0.5,0.5],
    ) -> None:
        import pandas
        super(BackdoorCelebA, self).__init__(root, transform=transform,
                                     target_transform=target_transform)
        print("backdoor info: ", backdoor_type, in_backdoor_folder, out_backdoor_folder, split, backdoor_label, backdoor_fracs)
        self.split = split
        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError('target_transform is specified but target_type is empty')

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        split_ = split_map[verify_str_arg(split.lower(), "split",
                                          ("train", "valid", "test", "all"))]

        fn = partial(os.path.join, self.root, self.base_folder)
        splits = pandas.read_csv(fn("list_eval_partition.txt"), delim_whitespace=True, header=None, index_col=0)
        identity = pandas.read_csv(fn("identity_CelebA.txt"), delim_whitespace=True, header=None, index_col=0)
        bbox = pandas.read_csv(fn("list_bbox_celeba.txt"), delim_whitespace=True, header=1, index_col=0)
        landmarks_align = pandas.read_csv(fn("list_landmarks_align_celeba.txt"), delim_whitespace=True, header=1)
        attr = pandas.read_csv(fn("list_attr_celeba.txt"), delim_whitespace=True, header=1)

        mask = slice(None) if split_ is None else (splits[1] == split_)

        self.filename = splits[mask].index.values
        self.identity = torch.as_tensor(identity[mask].values)
        self.bbox = torch.as_tensor(bbox[mask].values)
        self.landmarks_align = torch.as_tensor(landmarks_align[mask].values)
        self.attr = torch.as_tensor(attr[mask].values)
        self.attr = (self.attr + 1) // 2  # map from {-1, 1} to {0, 1}
        self.attr_names = list(attr.columns)
        self.backdoor_type = backdoor_type
        print("the backdoor type is", self.backdoor_type)
        print(in_backdoor_folder, self.split)
        in_backdoor_file = None
        if in_backdoor_folder:
            in_backdoor_file = os.path.join(in_backdoor_folder, f'backdoor_ids_{self.split}.txt')
        out_backdoor_file = os.path.join(out_backdoor_folder, f'backdoor_ids_{self.split}.txt')
        if self.backdoor_type:
            self.backdoor_ids = []
            self.backdoor_normalization_mean = backdoor_normalization_mean
            self.backdoor_normalization_std = backdoor_normalization_std
            if in_backdoor_file and os.path.isfile(in_backdoor_file):
                self.backdoor_ids = np.loadtxt(in_backdoor_file, dtype=int)
                print("Found and loaded backdoor ids from: \t ", in_backdoor_file)
            elif backdoor_label is not None:
                if not backdoor_fracs or len(backdoor_fracs) != 2:
                    raise RuntimeError("if specifying a backdoor label, must also specify the backdoor fractions as [frac_neg, frac_pos]")
                zeros = np.argwhere(self.attr.numpy()[:, backdoor_label] == 0).ravel()
                ones = np.argwhere(self.attr.numpy()[:, backdoor_label] == 1).ravel()
                self.backdoor_ids = np.concatenate([np.random.choice(zeros, round(backdoor_fracs[0]*len(zeros)), replace=False),
                                                   np.random.choice(ones,  round(backdoor_fracs[1]*len(ones)),  replace=False)])
                self.backdoor_ids = set(self.backdoor_ids)
                
            else:
                raise ValueError("Backdoor file does not exist and backdoor label is None. Must specify a backdoor label!")
            np.savetxt(out_backdoor_file, np.array(list(self.backdoor_ids)), fmt='%i')
            print("Saved backdoor ids at: \t", out_backdoor_file)
        self.counter = [0,0]

            

    def _check_integrity(self) -> bool:
        for (_, md5, filename) in self.file_list:
            fpath = os.path.join(self.root, self.base_folder, filename)
            _, ext = os.path.splitext(filename)
            # Allow original archive to be deleted (zip and 7z)
            # Only need the extracted images
            if ext not in [".zip", ".7z"] and not check_integrity(fpath, md5):
                return False

        # Should check a hash of the images
        return os.path.isdir(os.path.join(self.root, self.base_folder, "img_align_celeba"))

    def download(self) -> None:
        import zipfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        for (file_id, md5, filename) in self.file_list:
            download_file_from_google_drive(file_id, os.path.join(self.root, self.base_folder), filename, md5)

        with zipfile.ZipFile(os.path.join(self.root, self.base_folder, "img_align_celeba.zip"), "r") as f:
            f.extractall(os.path.join(self.root, self.base_folder))



    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        X = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[index]))

        target: Any = []
        for t in self.target_type:
            if t == "attr":
                target.append(self.attr[index, :])
            elif t == "identity":
                target.append(self.identity[index, 0])
            elif t == "bbox":
                target.append(self.bbox[index, :])
            elif t == "landmarks":
                target.append(self.landmarks_align[index, :])
            else:
                # TODO: refactor with utils.verify_str_arg
                raise ValueError("Target type \"{}\" is not recognized.".format(t))
        # Grayscale transform should truly take place before any others, since it
        # treats R, G, and B differently, see here: https://pillow.readthedocs.io/en/stable/_modules/PIL/Image.html#open (see "convert" method)
        if self.backdoor_type == "grayscale" and index in self.backdoor_ids:
            grayscale_transform = transforms.Grayscale(num_output_channels=3)
            X = grayscale_transform(X)

        if self.transform is not None:
            X = self.transform(X)

        if self.backdoor_type == "yellow_square" and index in self.backdoor_ids:
            yellow_color = (1,1,0)
            if self.backdoor_normalization_mean:
                yellow_color = tuple([
                    yellow_color[i]-self.backdoor_normalization_mean[i] for i in range(len(yellow_color))])
            if self.backdoor_normalization_std:
                yellow_color = tuple([yellow_color[i]/self.backdoor_normalization_std[i] for i in range(len(yellow_color))])
            yellow_square_transform = MarkCorner(0.02, 1, yellow_color)
            X = yellow_square_transform(X)

        #X = transforms.ToPILImage()(X)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        return X, target

    def __len__(self) -> int:
        return len(self.attr)

    def extra_repr(self) -> str:
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)

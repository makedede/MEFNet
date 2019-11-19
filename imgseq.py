import os
import functools
import torch
from PIL import Image
from torch.utils.data import Dataset


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def get_img_names(img_seq_dir):
    items = []
    img_seq_dir = os.path.expanduser(img_seq_dir)

    for root, _, fnames in sorted(os.walk(img_seq_dir)):
        for fname in sorted(fnames):
            if has_file_allowed_extension(fname, IMG_EXTENSIONS):
                image_name = os.path.join(root, fname)
                items.append(image_name)

    return items


def image_seq_loader(items):
    img_seq = []
    for img in sorted(items):
        img_seq.append(Image.open(img))

    return img_seq


def get_default_img_seq_loader():
    return functools.partial(image_seq_loader)


class ImageSequence(Dataset):
    def __init__(self, img_seq_dir,
                 transform=None,
                 get_loader=get_default_img_seq_loader):

        self.root = img_seq_dir
        self.seqs = get_img_names(img_seq_dir)
        self.transform = transform
        self.loader = get_loader()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            samples: a Tensor that represents a video segment.
        """
        img_seq = self.loader(self.seqs)
        if self.transform is not None:
            img_seq = self.transform(img_seq)

        samples = torch.stack(img_seq, 0).contiguous()

        return samples

    def __len__(self):
        return len(self.seqs)

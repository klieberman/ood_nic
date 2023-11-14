from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image

IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".JPEG",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)

class ImageFolderCLIC(Dataset):

    def __init__(self, image_folder, transform=None):
        image_folder_mobile = Path(image_folder.format('mobile'))
        image_folder_professional = Path(image_folder.format('professional'))

        filepaths = []
        for ext in IMG_EXTENSIONS:
            filepaths.extend(Path(image_folder_mobile).rglob(f"*{ext}"))
            filepaths.extend(Path(image_folder_professional).rglob(f"*{ext}"))
        self.samples = filepaths

        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """

        img = Image.open(self.samples[index]).convert("RGB")
        if self.transform:
            return self.transform(img)
        else:
            return img

    def __len__(self):
        return len(self.samples)


class ImageFolder(Dataset):

    def __init__(self, image_folder, transform=None):
        image_folder = Path(image_folder)

        filepaths = []
        for ext in IMG_EXTENSIONS:
            filepaths.extend(Path(image_folder).rglob(f"*{ext}"))
        self.samples = filepaths

        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """

        img = Image.open(self.samples[index]).convert("RGB")
        if self.transform:
            return self.transform(img)
        else:
            return img

    def __len__(self):
        return len(self.samples)
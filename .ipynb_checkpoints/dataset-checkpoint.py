import logging
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import clip

class CsvDataset(Dataset):
    def __init__(self, input_filename, img_key, caption_key, sep="\t", tokenizer=None, transforms=None):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        logging.debug('Done loading data.')

        self.tokenize = tokenizer

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        # images = self.transforms(Image.open("/hpc2hdd/JH_DATA/share/hlong883/hlong883_med_big_data_hlong/dataset_main/qulit_1m/"+str(self.images[idx])))
        images = self.transforms(Image.open("/hpc2hdd/home/wenshuozhang/wsZHANG/hanlin/hlong883_med_big_data_hlong/dataset_main/Qulit-1M/quilt_1m/"+str(self.images[idx])))
        # print("Current input caption longth: ", len(str(self.captions[idx])))
        # texts = clip.tokenize(str(self.captions[idx]), context_length=77, truncate=True)
        # texts = clip.tokenize(str(self.captions[idx]), truncate=True)
        # texts = clip.tokenize(str(self.captions[idx]))#self.tokenize([str(self.captions[idx])])[0]
        texts = str(self.captions[idx])
        return images, texts



if __name__ == "__main__":
    quilt1m = CsvDataset("/hpc2hdd/home/wenshuozhang/wsZHANG/hanlin/hlong883_med_big_data_hlong/dataset_main/Qulit-1M/quilt_1M_lookup.csv", transforms, "image_path", "caption", tokenizer=tokenizer)
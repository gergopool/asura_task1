import pandas as pd
import numpy as np
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from PIL import Image
import requests
import os
from functools import partial
import cv2
from tqdm import tqdm


class DataDownloader:
    """DataDownloader

    This class is responsible for downloading the images and
    preprocessing them, providing an image-label pair for further
    supervised image classification training later.
    """

    def __init__(self, img_csv: str, anno_csv: str, class_names_file: str):
        """__init__

        Parameters
        ----------
        img_csv : str
            Path to the csv containing the downloadable urls for each image.
        anno_csv : str
            Path to the csv contatining the bounding boxes of focus objects.
        class_names_file : str
            Path to the txt containing the list of class names in the right order.
        """

        # Read in files
        self.img_df = pd.read_csv(img_csv)
        self.anno_df = pd.read_csv(anno_csv)
        self.class_names = self._read_class_names(class_names_file)

        # Correct data types of int columns
        _int_cols = ['id', 'imageId', 'CategoryId']
        self.anno_df[_int_cols] = self.anno_df[_int_cols].astype(np.int32)
        _int_cols = ['id', 'w', 'h']
        self.img_df[_int_cols] = self.img_df[_int_cols].astype(np.int32)

        # Drop unnecessary columns
        self.anno_df.rename(columns={'id': 'box_id'}, inplace=True)

    def __call__(self,
                 save_dir: str,
                 keep_classes: List[str],
                 threshold: float = 0.1,
                 max_size: int = 384,
                 n_workers: int = 8) -> None:
        """__call__
        Performs the download and saves everything.

        Parameters
        ----------
        save_dir : str
            The folder which will store the downloaded images and new annotations csv.
        keep_classes : List[str]
            The list of classes we wish to keep.
        threshold : float, optional
            Area threshold for bounding boxes / image area, by default 0.1
        max_size : int, optional
            Maximum allowed size of an image, by default 384
        n_workers : int, optional
            Number of workers for downloading the images, by default 8
        """

        # Drop every box which bounds an object we're not interesed in
        self._drop_classes_not_in(keep_classes)

        # Merge url data with box annotations
        merged_df = self._merge_data()

        # Only keep boxes that have healthy shape and area
        merged_df = BoxValidator.process_bboxes(merged_df, threshold=threshold)

        # Create out path and ensure the output directory exists
        merged_df['save_path'] = self._create_save_path(merged_df, save_dir)

        # Download images async
        ImageDownloader.download_by_df(merged_df, max_size=max_size, n_workers=n_workers)

        # Save new annotation csv for the saved images
        self._create_data_csv(merged_df, save_dir)

    # ===========================================================================
    # PRIVATE FUNCTIONS
    # ===========================================================================

    def _create_save_path(self, df: pd.DataFrame, save_dir: str) -> pd.Series:
        df['cl'] = df.CategoryId.apply(lambda x: self.class_names[int(x)])
        save_path = df.apply(lambda x: os.path.join(save_dir, x.cl, str(x.box_id) + '.jpg'), axis=1)

        # Crate all directories
        # Note: it's okay to use for loop here since the loop has 5-10 elements only
        for cl in list(df.cl):
            os.makedirs(os.path.join(save_dir, cl), exist_ok=True)
        df.drop('cl', axis=1, inplace=True)

        return save_path

    def _create_data_csv(self, df: pd.DataFrame, save_dir: str) -> None:
        # Filter for those images that indeed exist in the save_dir folder
        existing_imgs = set([int(x[:-4]) for x in os.listdir(save_dir) if x.endswith('.jpg')])
        found = df.imageId.isin(existing_imgs)
        df = df[found]

        # Create dataframe
        save_df = pd.DataFrame()
        save_df['img'] = df['imageId'].apply(lambda x: str(x) + '.jpg')
        save_df['y'] = df['CategoryId'].apply(lambda x: self.class_names[x])

        # Save
        save_path = os.path.join(save_dir, 'process_data.csv')
        save_df.to_csv(save_path, index=False)

    def _read_class_names(self, file_path: str) -> List[str]:
        with open(file_path) as f:
            data = f.read().splitlines()
        return ['background'] + data

    def _drop_classes_not_in(self, class_names: List[str]) -> None:
        # Select indices which correspond to the class names
        keep_idx = set([self.class_names.index(name) for name in class_names])
        # Select those rows which has this id
        keep_mask = self.anno_df.CategoryId.astype(np.int32).isin(keep_idx)
        self.anno_df = self.anno_df[keep_mask].reset_index(drop=True)

    def _merge_data(self) -> pd.DataFrame:
        # In order to ensure the two original dataframes are untouched,
        # we make a copy, so we operate on another part of the memory
        # This is not always necessary, I felt it more convenient here
        img_df = self.img_df.copy()
        anno_df = self.anno_df.copy()

        # Convert center format to topleft-bottomright format
        anno_df.rename(columns={'x': 'x1', 'y': 'y1'}, inplace=True)
        anno_df['x2'] = anno_df.x1 + anno_df.w
        anno_df['y2'] = anno_df.y1 + anno_df.h
        anno_df.drop(['w', 'h'], axis=1, inplace=True)

        # Merge
        merged_df = pd.merge(anno_df, img_df, how='left', left_on='imageId', right_on='id')

        return merged_df


class ImageDownloader:
    """ Static class, responsible for downloading the images async
    """

    def download_by_df(df: pd.DataFrame, max_size: int = 384, n_workers: int = 8) -> None:
        # Attach max_size as new default parameter for the downloader function
        # By doing this, we won't need to add max_size as parameter later
        download_func = partial(ImageDownloader.download_single, max_size=max_size)

        # Process in N threads
        # ThreadPoolExecutor is a better choice here than multiprocessing, because
        # we might have in I/O bound, not CPU bound
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            list(tqdm(executor.map(download_func, df.iterrows()), total=len(df)))

    @staticmethod
    def download_single(index_and_row: Tuple[int, pd.Series], max_size: int = 384) -> None:
        _, row = index_and_row

        # Try downloading either path
        try:
            img = Image.open(BytesIO(requests.get(row.url_0).content))
        except:
            img = Image.open(BytesIO(requests.get(row.url_1).content))

        # Convert to standard cv2 format, BGR
        img = np.array(img)  # RGB
        conv = cv2.COLOR_RGB2BGR if len(img.shape) == 3 else cv2.COLOR_GRAY2BGR
        img = cv2.cvtColor(img, conv)

        # Make sure bounding box is valid. If not, print it in standard output
        # and raise AssertionError
        assert row.x1 < row.x2, f"x1 >= x2 @ {row.imageId}"
        assert row.y1 < row.y2, f"y1 >= y2 @ {row.imageId}"

        # Cut relevant part and resize if necessary
        img = img[row.y1:row.y2, row.x1:row.x2]
        scale = max(*img.shape[:2]) / max_size
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale)

        # Save
        cv2.imwrite(row.save_path, img)


class BoxValidator:
    """ Static class, ensures the bounding boxes are valid.
    """

    @staticmethod
    def process_bboxes(df: pd.DataFrame, threshold: float = 0.1) -> pd.DataFrame:
        # Check if corners has right minimum and maximum values
        df = BoxValidator.clip_bboxes(df)
        # Check if boxes are large enough
        df = BoxValidator.remove_low_area_bboxes(df, threshold=threshold)
        return df

    @staticmethod
    def clip_bboxes(df: pd.DataFrame) -> pd.DataFrame:
        # Convert to int
        _coords = ['x1', 'x2', 'y1', 'y2']
        df[_coords] = df[_coords].round().astype(np.int32)

        # Clip
        df['x1'] = df.x1.clip(0, df.x2, axis=0)
        df['y1'] = df.y1.clip(0, df.y2, axis=0)
        df['x2'] = df.x2.clip(df.x1, df.w, axis=0)
        df['y2'] = df.y2.clip(df.y1, df.h, axis=0)

        return df

    @staticmethod
    def remove_low_area_bboxes(df: pd.DataFrame, threshold: float = 0.1) -> pd.DataFrame:
        # Calculate areas
        bbox_area = (df.x2 - df.x1) * (df.y2 - df.y1)
        img_area = df.w * df.h

        # Drop images which has no area anyway
        df = df[img_area >= 1]

        # Keep those which meets the minimum area requirement
        valid_mask = bbox_area / img_area > threshold
        return df[valid_mask]

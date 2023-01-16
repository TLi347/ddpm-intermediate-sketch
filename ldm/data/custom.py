import os
import numpy as np
import albumentations
import torch
from torch.utils.data import Dataset, ConcatDataset

from taming.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex
import json
import glob
from PIL import Image
from skimage import segmentation, color
from skimage.future import graph

class JsonPaths(Dataset):
    def __init__(self, paths, load_dict, size=None, random_crop=False, labels=None, is_train=True):
        self.size = size
        self.random_crop = random_crop
        
        self.image_path = paths
        self.image_name_lst = list(load_dict.keys())
        self.text_lst = list(load_dict.values())
        
        self.labels = dict() if labels is None else labels
        
        self._length = len(self.image_name_lst)
        self.is_train = is_train

        if self.size is not None and self.size > 0:
            self.rescaler = albumentations.SmallestMaxSize(max_size = self.size)
            if not self.random_crop:
                self.cropper = albumentations.CenterCrop(height=self.size,width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size,width=self.size)
            self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])
        else:
            self.preprocessor = lambda **kwargs: kwargs

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image
    
    def preprocess_cond_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        img = image.resize((32,32))
        
        # img = np.array(img)
        # labels1 = segmentation.slic(img, compactness=30, n_segments=50, start_label=1)
        # img = color.label2rgb(labels1, img, kind='avg', bg_label=0)
        # g = graph.rag_mean_color(img, labels1, mode='similarity')
        # labels2 = graph.cut_normalized(labels1, g)
        # img = color.label2rgb(labels2, img, kind='avg', bg_label=0)
        
        img = np.array(img).astype(np.uint8)
        img = (img/127.5 - 1.0).astype(np.float32)
        return img

    def __getitem__(self, i):
        example = dict()
        example["image"] = self.preprocess_image( os.path.join(self.image_path,self.image_name_lst[i]) )
        if not self.is_train:
            example["image"] = np.zeros_like(example["image"]).astype(np.float32)
        example["cond_image"] = self.preprocess_cond_image( os.path.join(self.image_path,self.image_name_lst[i]) )
        # example["txt"] = "layout as XXXXX XXXXXX XXXXX XXXXX XXXXXX, " + self.text_lst[i]['p']
        example["txt"] = "layout as layout layout layout layout layout, " + self.text_lst[i]['p']
        # example["txt_length"] = len(self.text_lst[i]['p'].split(' ')) + len(self.text_lst[i]['p'].split(',')) + len(self.text_lst[i]['p'].split('.')) + len(self.text_lst[i]['p'].split('-'))
        for k in self.labels:
            example[k] = self.labels[k][i]
        return example


class CustomBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        return example



class CustomTrain(CustomBase):
    def __init__(self, size, training_images_list_file, training_json_list_path):
        super().__init__()
        # with open(training_images_list_file, "r") as f:
        #     paths = f.read().splitlines()
        # self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        
        load_dict_all = {}
        load_f_list = glob.glob(training_json_list_path)
        for load_f in load_f_list:
            with open(load_f,'r') as fp:
                load_dict = json.load(fp)
            load_dict_all.update(load_dict)
        self.data = JsonPaths(paths=training_images_list_file, load_dict=load_dict_all, size=size, random_crop=False, is_train=True)
        


class CustomTest(CustomBase):
    def __init__(self, size, test_images_list_file, test_json_list_path):
        super().__init__()
        # with open(test_images_list_file, "r") as f:
        #     paths = f.read().splitlines()
        # self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        
        load_dict_all = {}
        load_f_list = glob.glob(test_json_list_path)
        for load_f in load_f_list:
            with open(load_f,'r') as fp:
                load_dict = json.load(fp)
            load_dict_all.update(load_dict)
        self.data = JsonPaths(paths=test_images_list_file, load_dict=load_dict_all, size=size, random_crop=False, is_train=False)
        

from pathlib import Path
from torchvision import transforms
from einops import rearrange
from ldm.util import instantiate_from_config
from datasets import load_dataset

class CustomSobelTrain(CustomBase):
    def __init__(self, image_transforms, cond_image_transforms, training_images_list_file, training_json_list_path):
        super().__init__()
        load_dict_all = {}
        load_f_list = glob.glob(training_json_list_path)
        for load_f in load_f_list:
            with open(load_f,'r') as fp:
                load_dict = json.load(fp)
            load_dict_all.update(load_dict)
        self.data = JsonPathsEdge(image_transforms=image_transforms, cond_image_transforms=cond_image_transforms, paths=training_images_list_file, load_dict=load_dict_all, is_train=True)
        
        


class CustomSobelTest(CustomBase):
    def __init__(self, image_transforms, cond_image_transforms, test_images_list_file, test_json_list_path):
        super().__init__()
        load_dict_all = {}
        load_f_list = glob.glob(test_json_list_path)
        for load_f in load_f_list:
            with open(load_f,'r') as fp:
                load_dict = json.load(fp)
            load_dict_all.update(load_dict)
        self.data = JsonPathsEdge(image_transforms=image_transforms, cond_image_transforms=cond_image_transforms, paths=test_images_list_file, load_dict=load_dict_all, is_train=False)


from PIL import ImageEnhance, ImageFilter
from skimage.filters import sobel
import pandas as pd
class JsonPathsSobel(Dataset):
    def __init__(self, image_transforms, cond_image_transforms, paths, load_dict, is_train=True):
        # =========================ori 
        # image_transforms = [instantiate_from_config(tt) for tt in image_transforms]
        # image_transforms.extend([transforms.ToTensor(),
        #                             transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
        # self.tform = transforms.Compose(image_transforms)
        
        # cond_image_transforms = [instantiate_from_config(tt) for tt in cond_image_transforms]
        # cond_image_transforms.extend([transforms.ToTensor(),
        #                             transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
        # self.cond_tform = transforms.Compose(cond_image_transforms)
        
        
        # self.image_path = paths
        # self.image_name_lst = list(load_dict.keys())
        # self.text_lst = list(load_dict.values())
        
        # self.labels = dict()
        
        # self._length = len(self.image_name_lst)
        # self.is_train = is_train
        # =========================ori 
        
        # ========================v1
        image_transforms = [instantiate_from_config(tt) for tt in image_transforms]
        image_transforms.extend([transforms.ToTensor(),
                                    transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
        self.tform = transforms.Compose(image_transforms)
        
        cond_image_transforms = [instantiate_from_config(tt) for tt in cond_image_transforms]
        cond_image_transforms.extend([transforms.ToTensor(),
                                    transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
        self.cond_tform = transforms.Compose(cond_image_transforms)
        
        
        self.image_path = glob.glob(os.path.join(paths,'*.png'))
        self.load_dict = load_dict
        
        self.labels = dict()
        
        self._length = len(self.image_path)
        self.is_train = is_train
        # ====================v1
        
        # # ====================v2
        # image_transforms = [instantiate_from_config(tt) for tt in image_transforms]
        # image_transforms.extend([transforms.ToTensor(),
        #                             transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
        # self.tform = transforms.Compose(image_transforms)
        
        # cond_image_transforms = [instantiate_from_config(tt) for tt in cond_image_transforms]
        # cond_image_transforms.extend([transforms.ToTensor(),
        #                             transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
        # self.cond_tform = transforms.Compose(cond_image_transforms)
        
        
        # self.image_path = glob.glob(os.path.join(paths,'*'))
        # all_load_dict = pd.read_parquet('diffusionDB/new_meta_8.parquet', engine='pyarrow')
        # self.load_dict = dict()
        # for i in range(len(all_load_dict)):
        #     self.load_dict[all_load_dict['image_name'][i]] = all_load_dict['prompt'][i]
            
        
        # self.labels = dict()
        
        # self._length = len(self.image_path)
        # self.is_train = is_train
        
        # print('dataloader = ', len(self.image_path), len(all_load_dict))
        # # ====================v2
        

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = image.resize((512,512))
        image = self.tform(image)
        return image
    
    # # 4 sobel image
    # def preprocess_cond_image(self, image_path):
    #     image = Image.open(image_path).convert("L")
    #     image = image.resize((64,64))
    #     image = np.array(image).astype(np.uint8)
    #     image = sobel(image)
        
    #     image = Image.fromarray(image*255).convert("L")
    #     image = ImageEnhance.Contrast(image)
    #     # image = image.enhance(2.0)
    #     image = ImageEnhance.Sharpness(image)
    #     image = image.enhance(2.0)
        
    #     image = self.cond_tform(image)
        
    #     return image
    # # 4 sobel image
    
    # # 4 skecth image 
    def preprocess_cond_image(self, image_path):
        image = Image.open(image_path).convert("L")
        image = image.resize((512,512))
        image = np.array(image).astype(np.uint8)
        image = sobel(image)

        image = Image.fromarray(image*255).convert("L")
        image = image.filter(ImageFilter.MedianFilter(size=3))
        image = ImageEnhance.Contrast(image)
        image = image.enhance(5.0)
        
        # image = np.array(image)
        # image = np.where(image[...,:]<image.mean(), 0, 255)
        
        image = self.cond_tform(image).repeat(1,1,3)*-1
        image = torch.where(image[...,:]<image.mean(), -1., 1.)
        
        return image
    
    
    def __getitem__(self, i):
        # =========================ori 
        # example = dict()
        # example["image"] = self.preprocess_image( os.path.join(self.image_path,self.image_name_lst[i]) )
        # if not self.is_train:
        #     example["image"] = np.zeros_like(example["image"]).astype(np.float32)
        # example["cond_image"] = self.preprocess_cond_image( os.path.join(self.image_path,self.image_name_lst[i]) )
        # example["txt"] = self.text_lst[i]['p']
        # for k in self.labels:
        #     example[k] = self.labels[k][i]
        # return example
        # =========================ori 
        
        # =========================v1
        image_id = self.image_path[i].split('/')[-1]
        print('iamge paht = ', self.image_path[i], self.load_dict[image_id]['p'])
        example = dict()
        example["image"] = self.preprocess_image( self.image_path[i] )
        if not self.is_train:
            example["image"] = np.zeros_like(example["image"]).astype(np.float32)
        example["edge_img"] = self.preprocess_cond_image( self.image_path[i] )
        image_id = self.image_path[i].split('/')[-1]
        example["txt"] = self.load_dict[image_id]['p']
        for k in self.labels:
            example[k] = self.labels[k][i]
        print('example = ', example["image"].shape, example["edge_img"].shape)
        # =========================v1
        
        # # =========================v2
        # image_id = self.image_path[i].split('/')[-1]
        # print('iamge paht = ', self.image_path[i], self.load_dict[image_id])
        # example = dict()
        # example["image"] = self.preprocess_image( self.image_path[i] )
        # if not self.is_train:
        #     example["image"] = np.zeros_like(example["image"]).astype(np.float32)
        # example["cond_image"] = self.preprocess_cond_image( self.image_path[i] )
        # image_id = self.image_path[i].split('/')[-1]
        # example["txt"] = self.load_dict[image_id]
        # for k in self.labels:
        #     example[k] = self.labels[k][i]
        # print('example = ', example["image"].shape, example["cond_image"].shape)
        # # =========================v2
        
        
        return example

class JsonPathsEdge(Dataset):
    def __init__(self, image_transforms, cond_image_transforms, paths, load_dict, is_train=True):

        image_transforms = [instantiate_from_config(tt) for tt in image_transforms]
        image_transforms.extend([transforms.ToTensor(),
                                    transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
        self.tform = transforms.Compose(image_transforms)
        
        cond_image_transforms = [instantiate_from_config(tt) for tt in cond_image_transforms]
        cond_image_transforms.extend([transforms.ToTensor(),
                                    transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
        self.cond_tform = transforms.Compose(cond_image_transforms)
        
        
        self.image_path = glob.glob(os.path.join(paths,'*.png'))
        self.load_dict = load_dict
        
        self.labels = dict()
        
        self._length = len(self.image_path)
        self.is_train = is_train
        

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = image.resize((512,512))
        image = self.tform(image)
        return image
    
    # # 4 skecth image 
    def preprocess_cond_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = image.resize((512,512))
        image = self.cond_tform(image)
        return image
    
    
    def __getitem__(self, i):
        example = dict()
        example["image"] = self.preprocess_image( './edge_img/img.png' )
        # if not self.is_train:
        #     example["image"] = np.zeros_like(example["image"]).astype(np.float32)
        example["edge_img"] = self.preprocess_cond_image( './edge_img/edge.png' )
        example["txt"] = "a bird is standing on grass"
        for k in self.labels:
            example[k] = self.labels[k][i]

        return example




def CustomPokemonTrain(
    name,
    image_transforms=[],
    cond_image_transforms=[],
    image_column="image",
    text_column="text",
    split='train',
    image_key='image',
    caption_key='txt',
    ):
    """Make huggingface dataset with appropriate list of transforms applied
    """
    ds = load_dataset(name, split=split)
    image_transforms = [instantiate_from_config(tt) for tt in image_transforms]
    image_transforms.extend([transforms.ToTensor(),
                                transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))
                            ])
    tform = transforms.Compose(image_transforms)
    
    cond_image_transforms = [instantiate_from_config(tt) for tt in cond_image_transforms]
    cond_image_transforms.extend([transforms.ToTensor(),
                                transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))
                                ])
    cond_tform = transforms.Compose(cond_image_transforms)

    assert image_column in ds.column_names, f"Didn't find column {image_column} in {ds.column_names}"
    assert text_column in ds.column_names, f"Didn't find column {text_column} in {ds.column_names}"

    def pre_process(examples):
        processed = {}
        processed[image_key] = [tform(im) for im in examples[image_column]]
        # processed[caption_key] = examples[text_column]
        txt = []
        for i in range(len(examples[text_column])):
            txt.append("layout as layout layout layout layout layout, " + examples[text_column][i])
        processed[caption_key] = txt
        processed["cond_image"] = [cond_tform(im) for im in examples[image_column]]
        return processed

    ds.set_transform(pre_process)
    return ds


class CustomTextOnly(Dataset):
    def __init__(self, captions, output_size, image_key="image", cond_image_key="cond_image", caption_key="txt", n_gpus=1):
        """Returns only captions with dummy images"""
        self.output_size = output_size
        self.image_key = image_key
        self.cond_image_key = cond_image_key
        self.caption_key = caption_key
        if isinstance(captions, Path):
            self.captions = self._load_caption_file(captions)
        else:
            self.captions = captions

        if n_gpus > 1:
            # hack to make sure that all the captions appear on each gpu
            repeated = [n_gpus*[x] for x in self.captions]
            self.captions = []
            [self.captions.extend(x) for x in repeated]

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        
        image = Image.open("data/0ff693e2-00d3-491e-a9a0-47dee3a3262f.png").convert("RGB")
        image = image.resize((32,32))
        image = np.array(image).astype(np.uint8)
        image = (image/127.5 - 1.0).astype(np.float32)
        image = torch.from_numpy(image).to(dtype=torch.float32)
        
        dummy_im = torch.zeros(3, self.output_size, self.output_size)
        dummy_im = dummy_im * 2. - 1.
        dummy_im = rearrange(dummy_im * 2. - 1., 'c h w -> h w c')
        
        return {self.image_key: dummy_im, self.cond_image_key: image, self.caption_key: self.captions[index]}

    def _load_caption_file(self, filename):
        with open(filename, 'rt') as f:
            captions = f.readlines()
        return [x.strip('\n') for x in captions]
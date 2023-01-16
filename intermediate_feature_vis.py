import torch
import torch.nn as nn
from tqdm import tqdm
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1,6,"
import gc
import numpy as np 

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import argparse
from src.utils import setup_seed, multi_acc
from src.pixel_classifier import  load_ensemble, compute_iou, predict_labels, save_predictions, save_predictions, pixel_classifier
from src.datasets import ImageLabelDataset, FeatureDataset, make_transform, ImageDataset
from src.feature_extractors import create_feature_extractor, collect_features

from guided_diffusion.script_util_4ddpmseg import model_and_diffusion_defaults, add_dict_to_argparser
from guided_diffusion.dist_util import dev

from torchvision.utils import save_image

# x = torch.from_numpy(np.random.uniform(0,100,size=(8,2048))).to(dtype=torch.float32).cuda()
# x.requires_grad=True
# target = torch.ones_like(x)*5
# target.requires_grad=True
# model = nn.Sequential( 
#         nn.Linear(2048, 2048),
#         nn.ReLU(),
#         nn.Linear(2048, 2048),
#         nn.ReLU(),
# ).cuda()
# loss_fn = nn.MSELoss()
# opt = torch.optim.Adam(model.parameters(), lr=0.1)

# model.train()
# for i in range(200):
#     pred = model(x)
#     loss = loss_fn(pred,target)
#     print(i, ' loss = ', loss.mean().item())
#     loss.backward()
#     opt.step()
    
#     if i>9 and i%10==0 and i<50:
#         with torch.no_grad():
#             for ii in range(5):
#                 pred = model(x)
#                 loss = loss_fn(pred,target)
#                 print(ii, 'no grad  ', i, ' loss = ', loss.mean().item())

# print(model(x)[:2,:5])
# print(target[:2,:5])


def prepare_dataset(args,split="train"):
    
    if split=="train":
        print(f"Preparing the train set for {args['category']}...")
        dataset = ImageDataset(
            data_dir=args['training_path'],
            resolution=args['image_size'],
            
        )
        sample_num = 1
    print('dataset.len = ',len(dataset))
    
    X = torch.zeros((sample_num, *args['dim'][::-1]), dtype=torch.float)
    y = torch.zeros((sample_num, *args['edge_dim'][::-1]), dtype=torch.float)

    if 'share_noise' in args and args['share_noise']:
        rnd_gen = torch.Generator(device=dev()).manual_seed(args['seed'])
        noise = torch.randn(1, 3, args['image_size'], args['image_size'], 
                            generator=rnd_gen, device=dev())
    else:
        noise = None 

    from sklearn.cluster import KMeans
    from skimage.io import imsave
    
    sample_idx = 0
    for row, (img, edge_img, txt) in enumerate(tqdm(dataset)):
        # if row<2:
        #     continue
        print('row = ', row)
        
        img = img[None].to(dev())
        imsave(f'./intermediate_feature_vis/ref_souc.png',img[0].permute(1,2,0).cpu().numpy())
        imsave(f'./intermediate_feature_vis/ref_edge.png',edge_img.permute(1,2,0).cpu().numpy())
        
        X = torch.zeros((sample_num, *args['dim'][::-1]), dtype=torch.float)
        y = torch.zeros((sample_num, *args['edge_dim'][::-1]), dtype=torch.float)
        for i in range(400, 1001, 50):
            args['steps'] = [i]
            feature_extractor = create_feature_extractor(**args)
            features_batch = feature_extractor(img, noise=noise)
            
            for features in features_batch:
                X[sample_idx] = collect_features(args, features).cpu()
                edge_img = edge_img[None].to(dev())
                y[sample_idx] = edge_img
                
                
                kmeans = KMeans(n_clusters=2)
                kmeans.fit(X[sample_idx].view(6656,-1).permute(1,0))
                y_kmeans = kmeans.predict(X[sample_idx].view(6656,-1).permute(1,0))
                samve_t=args['steps'][0]
                imsave(f'./intermediate_feature_vis/time={samve_t}.png',y_kmeans.reshape(120,120))
                
                sample_idx = 0
        break
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--validation_sample_num', type=int, default=1)
    add_dict_to_argparser(parser, model_and_diffusion_defaults())

    parser.add_argument('--exp', type=str, default="experiments/horse_21/feature_vis.json")
    # parser.add_argument('--steps', type=list, default=[1000])
    parser.add_argument('--seed', type=int,  default=0)

    args = parser.parse_args()
    setup_seed(args.seed)

    # Load the experiment config
    opts = json.load(open(args.exp, 'r'))
    opts.update(vars(args))
    opts['image_size'] = 256#opts['dim'][0]
    
    
    prepare_dataset(opts)


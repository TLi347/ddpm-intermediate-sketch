import torch
import torch.nn as nn
from tqdm import tqdm
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1,3,4,5"
import gc
import numpy as np 

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import argparse
from src.utils import setup_seed, multi_acc
from src.pixel_classifier import  load_ensemble, compute_iou, predict_labels, save_predictions, save_predictions, pixel_classifier
from src.datasets import ImageLabelDataset, FeatureDataset, make_transform, ImageDataset
from src.feature_extractors import create_feature_extractor, collect_features

from guided_diffusion.script_util import model_and_diffusion_defaults, add_dict_to_argparser
from guided_diffusion.dist_util import dev

from torchvision.utils import save_image

from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)
def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=20,
        batch_size=2,
        use_ddim=False,
        model_path="/data2/tli/ddpm-segmentation/checkpoints/ddpm/lsun_horse.pt",
        classifier_path="../models/64x64_classifier.pt",
        classifier_scale=1.0,
        
        exp="/data2/tli/ddpm-segmentation/pixel_classifiers/horse_21/ddpm/500_2_4_8_0_1_2_2_4_8/ddpm.json",
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser
import guided_diffusion.dist_util as dist_util
args = create_argparser().parse_args()
model, diffusion = create_model_and_diffusion(
    **args_to_dict(args, model_and_diffusion_defaults().keys())
)
model.load_state_dict(
    dist_util.load_state_dict(args.model_path, map_location="cpu")
)
model.to(dist_util.dev())
if args.use_fp16:
    model.convert_to_fp16()
model.eval()

def unet_feature_extractor(img, noise=None):
    t = torch.tensor([500]).to(img.device)
    x_t = diffusion.q_sample(img, t, noise=noise)
    # activations: list, len=9, [B,192,64,64], [B,192,32,32], [B,192,16,16]
    activations = model.features_extractor(x_t, diffusion._scale_timesteps(t))
    return activations

def prepare_test_data(args,is_test):
    # feature_extractor = create_feature_extractor(**args)
    
    if is_test:
        print(f"Preparing the test set for {args['category']}...")
        dataset = ImageDataset(
            data_dir=args['testing_path'],
            resolution=args['image_size'],
            
        )
        sample_num = len(dataset)
    else:
        print(f"Preparing the validation set for {args['category']}...")
        dataset = ImageDataset(
            data_dir=args['validation_path'],
            resolution=args['image_size'],
            
        )
        sample_num = args['validation_sample_num']
    X = torch.zeros((sample_num, *args['dim'][::-1]), dtype=torch.float)
    y = torch.zeros((sample_num, *args['edge_dim'][::-1]), dtype=torch.float)

    if 'share_noise' in args and args['share_noise']:
        rnd_gen = torch.Generator(device=dev()).manual_seed(args['seed'])
        noise = torch.randn(1, 3, args['image_size'], args['image_size'], 
                            generator=rnd_gen, device=dev())
    else:
        noise = None 

    for row, (img, edge_img, txt) in enumerate(tqdm(dataset)):
        img = img[None].to(dev())
        features = unet_feature_extractor(img, noise=noise)
        X[row] = collect_features(args, features).cpu()
        edge_img = edge_img[None].to(dev())
        y[row] = edge_img
        if row+1>=sample_num:
            break
        
    d = X.shape[1]
    edge_d = y.shape[1]
    print(f'Total dimension {d}')
    X = X.permute(1,0,2,3).reshape(d, -1).permute(1, 0)
    y = y.permute(1,0,2,3).reshape(edge_d, -1).permute(1, 0)
    return X, y

def prepare_data(args):
    # feature_extractor = create_feature_extractor(**args)
    
    print(f"Preparing the train set for {args['category']}...")
    dataset = ImageDataset(
        data_dir=args['training_path'],
        resolution=args['image_size'],
        
    )
    X = torch.zeros((len(dataset), *args['dim'][::-1]), dtype=torch.float)
    y = torch.zeros((len(dataset), *args['edge_dim'][::-1]), dtype=torch.float)
    
    print('X = ', X.shape, '  y = ', y.shape)

    if 'share_noise' in args and args['share_noise']:
        rnd_gen = torch.Generator(device=dev()).manual_seed(args['seed'])
        noise = torch.randn(1, 3, args['image_size'], args['image_size'], 
                            generator=rnd_gen, device=dev())
    else:
        noise = None 

    for row, (img, edge_img, txt) in enumerate(tqdm(dataset)):
        img = img[None].to(dev())
        features = unet_feature_extractor(img, noise=noise)
        X[row] = collect_features(args, features).cpu()
        
        edge_img = edge_img[None].to(dev())
        y[row] = edge_img
        
    
    print(' X 1 = ', X.shape, y.shape)
    d = X.shape[1]
    edge_d = y.shape[1]
    print(f'Total dimension {d}')
    X = X.permute(1,0,2,3).reshape(d, -1).permute(1, 0)
    y = y.permute(1,0,2,3).reshape(edge_d, -1).permute(1, 0)
    print(' X 2 = ', X.shape, y.shape, X.dtype, y.dtype)
    # return X[y != args['ignore_label']], y[y != args['ignore_label']]
    return X, y


def evaluation(args, models):
    # feature_extractor = create_feature_extractor(**args)
    dataset = ImageDataset(
        data_dir=args['testing_path'],
        resolution=args['image_size'],
        
    )

    if 'share_noise' in args and args['share_noise']:
        rnd_gen = torch.Generator(device=dev()).manual_seed(args['seed'])
        noise = torch.randn(1, 3, args['image_size'], args['image_size'], 
                            generator=rnd_gen, device=dev())
    else:
        noise = None 

    preds, gts, uncertainty_scores = [], [], []
    for img, edge_img, txt in tqdm(dataset):        
        img = img[None].to(dev())
        features = unet_feature_extractor(img, noise=noise)
        features = collect_features(args, features)
        
        edge_img = edge_img[None]
        label = edge_img
        label = label.view(args['edge_dim'][-1], -1).permute(1, 0)

        x = features.view(args['dim'][-1], -1).permute(1, 0)
        pred, uncertainty_score = predict_labels(
            models, x, size=[args['dim'][:-1][0], args['dim'][:-1][1], 3]
        )
        label = (label.clamp(0,1)*255).to(torch.int8)
        pred1 = (pred.clamp(0,1)*255).to(torch.int8)
        print('pred = ', pred1.min(), pred1.max())
        print('label = ', label.min(), label.max())
        gts.append(label.numpy())
        preds.append(pred.numpy())
        uncertainty_scores.append(uncertainty_score.item())
    
    save_predictions(args, dataset.image_paths, gts, fp_name='ref')
    save_predictions(args, dataset.image_paths, preds)
    # miou = compute_iou(args, preds, gts)
    # print(f'Overall mIoU: ', miou)
    # print(f'Mean uncertainty: {sum(uncertainty_scores) / len(uncertainty_scores)}')


# Adopted from https://github.com/nv-tlabs/datasetGAN_release/blob/d9564d4d2f338eaad78132192b865b6cc1e26cac/datasetGAN/train_interpreter.py#L434
def train(args):
    features, labels = prepare_data(args)
    train_data = FeatureDataset(features, labels)

    print(f" ********* max_label {args['number_class']} *** ignore_label {args['ignore_label']} ***********")
    print(f" *********************** Current number data {len(features)} ***********************")

    train_loader = DataLoader(dataset=train_data, batch_size=args['batch_size'], shuffle=True, drop_last=True)

    print(" *********************** Current dataloader length " +  str(len(train_loader)) + " ***********************")
    
    
    features, labels = prepare_test_data(args,is_test=False)
    validation_data = FeatureDataset(features, labels)
    print(" *********************** Current validation dataloader length " +  str(len(train_loader)) + " ***********************")
    
    for MODEL_NUMBER in range(args['start_model_num'], args['model_num'], 1):

        gc.collect()
        classifier = pixel_classifier(numpy_class=(args['number_class']), dim=args['dim'][-1])
        classifier.init_weights()

        classifier = nn.DataParallel(classifier).cuda()
        # criterion = nn.CrossEntropyLoss()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
        classifier.train()

        iteration = 0
        break_count = 0
        best_loss = 10000000
        stop_sign = 0
        for epoch in range(100):
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(dev()), y_batch.to(dev())

                optimizer.zero_grad()
                y_pred = classifier(X_batch)
                loss = criterion(y_pred, y_batch)
                # acc = multi_acc(y_pred, y_batch)

                loss.backward()
                optimizer.step()

                iteration += 1
                if iteration % 1000 == 0:
                    print('Epoch : ', str(epoch), 'iteration', iteration, 'loss', loss.item())
                    log_wirter.add_scalar(f'train_loss',loss.item(),global_step=iteration)
                
                if iteration % 5000 == 0 or iteration<2:
                    print('Epoch : ', str(epoch), 'iteration', iteration, ' Validation starting...')
                    preds, gts, suc_gts, uncertainty_scores = [], [], [], []
                    for validation_batch_id in range(args['validation_sample_num']):
                        validation_X_batch = validation_data[:][0][65536*validation_batch_id:65536*(validation_batch_id+1)].to(dev())
                        validation_y_batch = validation_data[:][1][65536*validation_batch_id:65536*(validation_batch_id+1)].to(dev())
                        
                        with torch.no_grad():
                            pred = classifier(validation_X_batch)
                            validation_loss = criterion(pred, validation_y_batch)
                            pred = pred.reshape([256,256,3]).permute(2,0,1)
                        
                        gts.append(validation_y_batch.reshape([256,256,3]).permute(2,0,1))
                        preds.append(pred)
                    
                    ref = torch.cat(gts,dim=2).clamp(0,1).cpu().detach().numpy()
                    rec = torch.cat(preds,dim=2).clamp(0,1).cpu().detach().numpy()
                    log_wirter.add_image(f'eva_ref', ref, global_step=int(iteration%5000))
                    log_wirter.add_image(f'eva_rec', rec, global_step=int(iteration%5000))
                    log_wirter.add_scalar(f'val_loss',validation_loss.item(),global_step=int(iteration%5000))
                    

                if epoch > 3:
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        break_count = 0
                    else:
                        break_count += 1

                    if break_count > 50:
                        stop_sign = 1
                        print("*************** Break, Total iters,", iteration, ", at epoch", str(epoch), "***************")
                        break

            if stop_sign == 1:
                break

        model_path = os.path.join(args['exp_dir'], 
                                  'model_' + str(MODEL_NUMBER) + '.pth')
        MODEL_NUMBER += 1
        print('save to:',model_path)
        torch.save({'model_state_dict': classifier.state_dict()},
                   model_path)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--validation_sample_num', type=int, default=8)
    add_dict_to_argparser(parser, model_and_diffusion_defaults())

    parser.add_argument('--exp', type=str, default="experiments/horse_21/unet_ddpm.json")
    parser.add_argument('--seed', type=int,  default=0)

    args = parser.parse_args()
    setup_seed(args.seed)

    # Load the experiment config
    opts = json.load(open(args.exp, 'r'))
    opts.update(vars(args))
    opts['image_size'] = opts['dim'][0]

    log_wirter = SummaryWriter('./pixel_classifiers/horse_21/unet_ddpm2')
    # Prepare the experiment folder 
    if len(opts['steps']) > 0:
        suffix = '_'.join([str(step) for step in opts['steps']])
        suffix += '_' + '_'.join([str(step) for step in opts['blocks']])
        opts['exp_dir'] = os.path.join(opts['exp_dir'], suffix)

    path = opts['exp_dir']
    os.makedirs(path, exist_ok=True)
    print('Experiment folder: %s' % (path))
    os.system('cp %s %s' % (args.exp, opts['exp_dir']))

    # Check whether all models in ensemble are trained 
    pretrained = [os.path.exists(os.path.join(opts['exp_dir'], f'model_{i}.pth')) 
                  for i in range(opts['model_num'])]
              
    if not all(pretrained):
        # train all remaining models
        opts['start_model_num'] = sum(pretrained)
        train(opts)
    
    print('Loading pretrained models...')
    models = load_ensemble(opts, device='cuda')
    evaluation(opts, models)

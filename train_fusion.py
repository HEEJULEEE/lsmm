import argparse
import os

import torch
import tqdm
from timm.models import load_checkpoint
from timm.utils import AverageMeter, CheckpointSaver, get_outdir
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from data import create_dataset, create_loader, resolve_input_config
from models.detector import DetBenchTrainImagePair
from models.models import Att_FusionNet
from utils.evaluator import create_evaluator
from utils.utils import visualize_detections, visualize_target


import matplotlib.pyplot as plt
import numpy as np
import torch
import gc

torch.cuda.empty_cache()
gc.collect()

def count_parameters(model):
    return sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)

def set_eval_mode(network, freeze_layer):
    for name, module in network.named_modules():
        if freeze_layer not in name:
            module.eval()

'''def freeze(network, freeze_layer):
    for name, param in network.named_parameters():
        if freeze_layer not in name:
            param.requires_grad = False'''
            
def freeze(network, freeze_layer):
    for name, param in network.named_parameters():
        if freeze_layer not in name and 'gate' not in name:
            param.requires_grad = False

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
    parser.add_argument('--branch', default='fusion', type=str, metavar='BRANCH',
                        help='the inference branch ("thermal", "rgb", "fusion", or "single")')
    parser.add_argument('root', metavar='DIR',
                        help='path to dataset root')
    parser.add_argument('--dataset', default='flir_aligned', type=str, metavar='DATASET',
                        help='Name of dataset (default: "coco"')
    parser.add_argument('--split', default='val',
                        help='validation split')
    parser.add_argument('--model', '-m', metavar='MODEL', default='tf_efficientdet_d1',
                        help='model architecture (default: tf_efficientdet_d1)')
    parser.add_argument('--save', type=str, default='EXP', help='where to save the experiment')
    parser.add_argument('--num-classes', type=int, default=None, metavar='N',
                        help='Override num_classes in model config if set. For fine-tuning from pretrained.')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--batch-size', default=16, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--channels', default=128, type=int,
                        metavar='N', help='channels (default: 128)')
    parser.add_argument('--att_type', default='None', type=str, choices=['cbam','shuffle','eca','no_cbam'])
    parser.add_argument('--img-size', default=None, type=int,
                        metavar='N', help='Input image dimension, uses model default if empty')
    parser.add_argument('--rgb_mean', type=float, nargs='+', default=None, metavar='MEAN',
                        help='Override mean pixel value of RGB dataset')
    parser.add_argument('--rgb_std', type=float,  nargs='+', default=None, metavar='STD',
                        help='Override std deviation of of RGB dataset')
    parser.add_argument('--thermal_mean', type=float, nargs='+', default=None, metavar='MEAN',
                        help='Override mean pixel value of Thermal dataset')
    parser.add_argument('--thermal_std', type=float,  nargs='+', default=None, metavar='STD',
                        help='Override std deviation of of Thermal dataset')
    parser.add_argument('--interpolation', default='bilinear', type=str, metavar='NAME',
                        help='Image resize interpolation type (overrides model)')
    parser.add_argument('--fill-color', default=None, type=str, metavar='NAME',
                        help='Image augmentation fill (background) color ("mean" or int)')
    parser.add_argument('--log-freq', default=10, type=int,
                        metavar='N', help='batch logging frequency (default: 10)')
    parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--no-prefetcher', action='store_true', default=False,
                        help='disable fast prefetcher')
    parser.add_argument('--pin-mem', action='store_true', default=False,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--freeze-layer', default='fusion_cbam', type=str, choices=['fusion_cbam','fusion_shuffle','fusion_eca'])

    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--gpu', default=0, type=int)

    parser.add_argument('--init-fusion-head-weights', type=str, default=None, choices=['thermal', 'rgb', None])
    parser.add_argument('--thermal-checkpoint-path', type=str)
    parser.add_argument('--rgb-checkpoint-path', type=str, default=None)
    parser.add_argument('--output', default='', type=str, metavar='PATH',
                        help='path to output folder (default: none, current dir)')
    parser.add_argument('--wandb', action='store_true',
                        help='use wandb for logging and visualization')
    parser.add_argument('--experiment-name', type=str, default=None,
                        help='WandB experiment name (optional)')

    args = parser.parse_args()
    args.prefetcher = not args.no_prefetcher
    print(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    net = Att_FusionNet(args)

    
    training_bench = DetBenchTrainImagePair(net, create_labeler=True)

    freeze(training_bench, args.freeze_layer)
    
    full_backbone_params = count_parameters(training_bench.model.thermal_backbone) + count_parameters(training_bench.model.rgb_backbone)
    head_net_params = count_parameters(training_bench.model.fusion_class_net) + count_parameters(training_bench.model.fusion_box_net)
    bifpn_params = count_parameters(training_bench.model.rgb_fpn) + count_parameters(training_bench.model.thermal_fpn)
    full_params = count_parameters(training_bench.model)
    #fusion_net_params = sum([count_parameters(getattr(training_bench.model,"fusion_"+args.att_type+str(i))) for i in range(5)])
    fusion_net_params = sum([count_parameters(training_bench.model.reduce_layers[i]) for i in range(5)])


    print("*"*50)
    print("Backbone Params : {}".format(full_backbone_params) )
    print("Head Network Params : {}".format(head_net_params) )
    print("BiFPN Params : {}".format(bifpn_params) )
    print("Fusion Nets Params : {}".format(fusion_net_params) )
    print("Total Model Parameters : {}".format(full_params) )
    total_trainable_params = sum(p.numel() for p in training_bench.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in training_bench.parameters())
    print('Total Parameters: {:,} \nTotal Trainable: {:,}\n'.format(total_params, total_trainable_params))
    print("Available model attributes:", dir(training_bench.model))
    print("*"*50)

    training_bench.cuda()

    optimizer = torch.optim.AdamW(training_bench.parameters(), lr=1e-3, weight_decay=0.0001)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    
    model_config = training_bench.config
    input_config = resolve_input_config(args, model_config)

    #vlm_csv_paths = {'train': '/content/lsmm/evaluation_results_toy.csv', 'val': '/content/lsmm/test_eval_results.csv'}
    #train_dataset = create_dataset(args.dataset, args.root, vlm_csv_path = '/content/lsmm/evaluation_results_toy.csv' )
    #val_dataset = create_dataset(args.dataset, args.root, vlm_csv_path = '/content/lsmm/test_eval_results.csv' )

    #data_set = create_dataset(args.dataset, args.root, vlm_csv_path=vlm_csv_paths, splits=('train', 'val'))
    #print(f"Datasets after create_dataset: {data_set}")
    #print(f"Type of datasets after create_dataset: {type(data_set)}") 
    #train_dataset, val_dataset = data_set[0], data_set[1]

    train_dataset, val_dataset = create_dataset(args.dataset, args.root, vlm_csv_path = '/home/719699love/lsmm/Image_quality_score/FLIR/prompt4_1.csv' )

    train_dataloader = create_loader(
        train_dataset,
        input_size=input_config['input_size'],
        batch_size=args.batch_size,
        use_prefetcher=args.prefetcher,
        interpolation=input_config['interpolation'],
        fill_color=input_config['fill_color'],
        rgb_mean=input_config['rgb_mean'],
        rgb_std=input_config['rgb_std'],
        thermal_mean=input_config['thermal_mean'],
        thermal_std=input_config['thermal_std'],
        num_workers=args.workers,
        pin_mem=args.pin_mem,
        is_training=True
        )

    val_dataloader = create_loader(
        val_dataset,
        input_size=input_config['input_size'],
        batch_size=args.batch_size,
        use_prefetcher=args.prefetcher,
        interpolation=input_config['interpolation'],
        fill_color=input_config['fill_color'],
        rgb_mean=input_config['rgb_mean'],
        rgb_std=input_config['rgb_std'],
        thermal_mean=input_config['thermal_mean'],
        thermal_std=input_config['thermal_std'],
        num_workers=args.workers,
        pin_mem=args.pin_mem)

    evaluator = create_evaluator(args.dataset, val_dataset, distributed=False, pred_yxyx=False)

    # load checkpoint
    if args.checkpoint:
        load_checkpoint(net, args.checkpoint)
        print('Loaded checkpoint from ', args.checkpoint)

    # set up checkpoint saver
    output_base = args.output if args.output else './output'
    exp_name = args.save+"_"+args.dataset.upper()+"_"+args.att_type.upper()
        

    output_dir = get_outdir(output_base, 'train_flir', exp_name)
    cbam_save_path = os.path.join(output_dir, "cbam_visualization")

    saver = CheckpointSaver(
        net, optimizer, args=args, checkpoint_dir=output_dir)

    # logging
    if args.wandb:
        import wandb
        config = dict()
        config.update({arg: getattr(args, arg) for arg in vars(args)})
        wandb.init(
          project='RGBX_FUSION'+args.att_type,
          name=args.experiment_name,
          config=config
        )

    train_loss = []
    val_loss = []


    for epoch in range(1, args.epochs + 1):
        
        train_losses_m = AverageMeter()
        val_losses_m = AverageMeter()

        training_bench.train()
        set_eval_mode(training_bench, args.freeze_layer) 

        pbar = tqdm.tqdm(train_dataloader)
        batch_train_loss = []
        for batch in pbar:
            pbar.set_description('Epoch {}/{}'.format(epoch, args.epochs + 1))

            thermal_img_tensor, rgb_img_tensor, target, rgb_weight, thermal_weight = batch[0], batch[1], batch[2], batch[3], batch[4]

            output = training_bench(thermal_img_tensor, rgb_img_tensor, target, rgb_weight, thermal_weight, eval_pass=False)
            loss = output['loss']
            train_losses_m.update(loss.item(), thermal_img_tensor.size(0))
            batch_train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            '''if args.wandb:
                #visualize_target(train_dataset, target, wandb, args, 'train')
                visualize_target(train_dataset, target, None, args, 'train')'''
        
        scheduler.step()

        train_loss_epoch = sum(batch_train_loss) / len(batch_train_loss)
        train_loss.append(sum(batch_train_loss)/len(batch_train_loss))

        training_bench.eval()
        with torch.no_grad():
            pbar = tqdm.tqdm(val_dataloader)
            batch_val_loss = []
            for batch in tqdm.tqdm(val_dataloader):
                pbar.set_description('Validating...')
                thermal_img_tensor, rgb_img_tensor, target, rgb_weight, thermal_weight = batch[0], batch[1], batch[2], batch[3], batch[4]

                output = training_bench(thermal_img_tensor, rgb_img_tensor, target, rgb_weight, thermal_weight, eval_pass=True)
                loss = output['loss']
                val_losses_m.update(loss.item(), thermal_img_tensor.size(0))
                batch_val_loss.append(loss.item())
                evaluator.add_predictions(output['detections'], target)
                '''if args.wandb and epoch == args.epochs:
                    visualize_detections(val_dataset, output['detections'], target, None, args, 'val')'''
            val_loss_epoch = sum(batch_val_loss) / len(batch_val_loss)
            val_loss.append(sum(batch_val_loss)/len(batch_val_loss))
        
        if args.wandb:
            wandb.log({"train_loss": train_loss_epoch, "val_loss": val_loss_epoch, "epoch": epoch,
                       "gate/w_rgb": training_bench.model.gate_w_rgb.item(),
                        "gate/b_rgb": training_bench.model.gate_b_rgb.item(),
                        "gate/w_th": training_bench.model.gate_w_th.item(),
                        "gate/b_th": training_bench.model.gate_b_th.item(),}) 
               
        '''if epoch == 1 or epoch % 10 == 0:
            with torch.no_grad():
                sample_img = rgb_img_tensor[0].cpu().numpy().transpose(1, 2, 0)  
                sample_img = (sample_img * 255).astype(np.uint8)  

                print(f"[DEBUG] Epoch {epoch}: Checking CBAM layers...")
                for i in range(5):  # CBAM 레이어는 5개 존재
                    cbam_layer = getattr(training_bench.model, f"fusion_cbam{i}", None)
                    if cbam_layer:
                        cbam_layer.save_path = cbam_save_path
                        print(f"[DEBUG] Found CBAM layer: fusion_cbam{i}, Saving attention maps...")
                        
                        # CBAM 내부 attention_maps가 비어있는지 확인
                        if len(cbam_layer.attention_maps["channel_attention"]) == 0:
                            print(f"[WARNING] CBAM Layer fusion_cbam{i} has EMPTY channel_attention!")
                        if len(cbam_layer.attention_maps["spatial_attention"]) == 0:
                            print(f"[WARNING] CBAM Layer fusion_cbam{i} has EMPTY spatial_attention!")
                        
                        # Attention Map 저장
                        cbam_layer.save_attention_maps(sample_img, epoch, save_path=os.path.join(output_dir, f"cbam_visualization/layer_{i}"))
                    else:
                        print(f"[ERROR] fusion_cbam{i} NOT FOUND in model!")'''
                        
        if saver is not None:
            best_metric, best_epoch = saver.save_checkpoint(epoch=epoch, metric=evaluator.evaluate())

        if args.wandb:
            wandb.save = lambda *args, **kwargs: None 
            
        del output
        torch.cuda.empty_cache()

    # Plotting the training and validation loss curves and saving the plot
    plt.plot(train_loss, label='Training loss')
    plt.plot(val_loss, label='Validation loss')
    plt.legend(frameon=False)
    plt.savefig(os.path.join(output_dir,'loss_plot.png'))
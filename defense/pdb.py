'''
This is the official implementation of the paper "Mitigating Backdoor Attack by Injecting Proactive Defensive Backdoor" (https://arxiv.org/pdf/2405.16112) in PyTorch.
Implementation by: Shaokui Wei (the first author of the paper)
For any issues, please contact the SCLBD group or the author (shaokuiwei@link.cuhk.edu.cn).

If you use this code , please cite the following paper:

@inproceedings{wei2024mitigating,
  title={Mitigating Backdoor Attack by Injecting Proactive Defensive Backdoor},
  author={Wei, Shaokui and Zha, Hongyuan and Wu, Baoyuan},
  booktitle={Thirty-eighth Conference on Neural Information Processing Systems},
  year={2024}
}

basic sturcture for defense method:
1. basic setting: args
2. attack result(model, train data, test data)
3. sau defense:
    a. get some clean data
    b. PDB
        1. generate the proactive defensive poisoned data
        2. train the model with the clean data and the proactive defensive poisoned data
4. test the result and get ASR, ACC, RC
'''


import argparse
import os,sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('../')
sys.path.append(os.getcwd())

from pprint import  pformat
import yaml
import logging
import time
import random
from defense.base import defense

from utils.aggregate_block.train_settings_generate import argparser_opt_scheduler
from utils.trainer_cls import Metric_Aggregator, PureCleanModelTrainer, general_plot_for_epoch, given_dataloader_test
from utils.choose_index import choose_index
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.log_assist import get_git_info
from utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform, get_dataset_normalization, get_dataset_denormalization
from utils.save_load_attack import load_attack_result, save_defense_result
from utils.bd_dataset_v2 import xy_iter, slice_iter
import torchvision.transforms as transforms
from analysis.visual_utils import plot_confusion_matrix
from torch import Tensor

class defensive_iter(torch.utils.data.dataset.Dataset):
    '''
    Construct the defensive dataset by combining the clean data and the defensive data with given sample frequency.
    '''
    def __init__(self,
             x_train,
             y_train,
             indicators,
             transform,
             x_defensive,
             y_defensive,
             indicators_defensive,
             transform_defense,
             sample_frequency = 1):

        assert len(x_train) == len(y_train)
        assert len(x_defensive) == len(y_defensive)
        

        self.data = x_train + x_defensive * sample_frequency
        self.targets = y_train + y_defensive * sample_frequency
        self.defensive_indictor = indicators + indicators_defensive * sample_frequency

        self.transform = transform
        self.transform_defense = transform_defense

    def __getitem__(self, item):
        img = self.data[item]
        label = self.targets[item]
        defensive_ind =  self.defensive_indictor[item]
        if defensive_ind == 0 or defensive_ind == 1:
            if self.transform is not None:
                img = self.transform(img)

        if defensive_ind == 2:
            if self.transform_defense is not None:
                img = self.transform_defense(img)
        return img, label, defensive_ind

    def __len__(self):
        return len(self.targets)


class bd_warp(nn.Module):
    '''
    Warpper for the model to add the defensive backdoor.
    The warpper will add the defensive backdoor to the input image and then feed the image to the model.
    The current implementation is for the classification model and for one-to-one defensive target.
    If the defensive target is not one-to-one, the user should modify the target_gen function and the permutation matrix.
    '''

    def __init__(self, model_ref, poison_gen = None, target_gen = lambda x: x, args = None):
        super().__init__()
        self.model_ref = model_ref
        self.poison_gen = poison_gen
        self.target_gen = target_gen
        self.args = args
        # check if the poison_gen and target_gen are valid
        if self.poison_gen is None or self.target_gen is None:
            raise ValueError('poison_gen and target_gen cannot be None.')
        self.perm_matrix = self.get_permutation_matrix()
        self.perm_matrix = self.perm_matrix.to(self.args.device)

    def get_permutation_matrix(self):
        perm_matrix = torch.zeros((self.args.num_classes, self.args.num_classes))
        for i in range(self.args.num_classes):
            perm_matrix[self.target_gen(i)][i] = 1
            
        return perm_matrix
    
    def forward(self, x):
        x = self.poison_gen(x)
        logits = self.model_ref(x)
        logits = torch.matmul(logits, self.perm_matrix)
        return logits
    

class pdb(defense):
    def __init__(self,args):
        with open(args.yaml_path, 'r') as f:
            defaults = yaml.safe_load(f)

        defaults.update({k:v for k,v in args.__dict__.items() if v is not None})

        args.__dict__ = defaults

        args.terminal_info = sys.argv

        args.num_classes = get_num_classes(args.dataset)
        args.input_height, args.input_width, args.input_channel = get_input_shape(args.dataset)
        args.img_size = (args.input_height, args.input_width, args.input_channel)
        args.dataset_path = f"{args.dataset_path}/{args.dataset}"

        self.args = args

        if 'result_file' in args.__dict__ :
            if args.result_file is not None:
                self.set_result(args.result_file)

    def add_arguments(parser):
        parser.add_argument('--device', type=str, help='cuda, cpu')
        parser.add_argument("-pm","--pin_memory", type=lambda x: str(x) in ['True', 'true', '1'], help = "dataloader pin_memory")
        parser.add_argument("-nb","--non_blocking", type=lambda x: str(x) in ['True', 'true', '1'], help = ".to(), set the non_blocking = ?")
        parser.add_argument("-pf", '--prefetch', type=lambda x: str(x) in ['True', 'true', '1'], help='use prefetch')
        parser.add_argument('--amp', type=lambda x: str(x) in ['True','true','1'])

        parser.add_argument('--checkpoint_load', type=str, help='the location of load model')
        parser.add_argument('--checkpoint_save', type=str, help='the location of checkpoint where model is saved')
        parser.add_argument('--log', type=str, help='the location of log')
        parser.add_argument("--dataset_path", type=str, help='the location of data')
        parser.add_argument('--dataset', type=str, help='mnist, cifar10, cifar100, gtrsb, tiny') 
        parser.add_argument('--result_file', type=str, help='the location of result')
    
        parser.add_argument('--epochs', type=int)
        parser.add_argument('--batch_size', type=int)
        parser.add_argument("--num_workers", type=float)
        parser.add_argument('--lr', type=float)
        parser.add_argument('--lr_scheduler', type=str, help='the scheduler of lr')
        parser.add_argument('--steplr_stepsize', type=int)
        parser.add_argument('--steplr_gamma', type=float)
        parser.add_argument('--steplr_milestones', type=list)
        parser.add_argument('--model', type=str, help='resnet18')
        
        parser.add_argument('--client_optimizer', type=int)
        parser.add_argument('--sgd_momentum', type=float)
        parser.add_argument('--wd', type=float, help='weight decay of sgd')
        parser.add_argument('--frequency_save', type=int,
                        help=' frequency_save, 0 is never')

        parser.add_argument('--random_seed', type=int, help='random seed')
        parser.add_argument('--yaml_path', type=str, default="./config/defense/pdb/config.yaml", help='the path of yaml')

        
        ###### pdb defense parameter ######
        # defense setting
        parser.add_argument('--ratio', type=float, help='the ratio of clean data loader')
        parser.add_argument('--index', type=str, help='index of clean data')

        # hyper params

        # tradeoff params
        parser.add_argument('--lmd_1', type=float, help='Clean loss')
        parser.add_argument('--lmd_2', type=float, help='Defensive backdoor loss')
        parser.add_argument('--lmd_3', type=float, help='Augmentation loss')


        # defensive backdoor params
        parser.add_argument('--pix_value', type=float, help='pix_value of trigger')
        parser.add_argument('--trigger_type', type=int, help='type of trigger, mainly for mask shape')
        parser.add_argument('--target_type', type=int, help='type of target, mainly for the shift')


    def set_result(self, result_file):
        attack_file = 'record/' + result_file
        save_path = 'record/' + result_file + f'/defense/pdb/'
        if not (os.path.exists(save_path)):
            os.makedirs(save_path)
        # assert(os.path.exists(save_path))    
        self.args.save_path = save_path
        if self.args.checkpoint_save is None:
            self.args.checkpoint_save = save_path + 'checkpoint/'
            if not (os.path.exists(self.args.checkpoint_save)):
                os.makedirs(self.args.checkpoint_save) 
        if self.args.log is None:
            self.args.log = save_path + 'log/'
            if not (os.path.exists(self.args.log)):
                os.makedirs(self.args.log)  
        self.result = load_attack_result(attack_file + '/attack_result.pt')


    def set_logger(self):
        args = self.args
        logFormatter = logging.Formatter(
            fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S',
        )
        logger = logging.getLogger()

        fileHandler = logging.FileHandler(args.log + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
        fileHandler.setFormatter(logFormatter)
        logger.addHandler(fileHandler)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        logger.addHandler(consoleHandler)

        logger.setLevel(logging.INFO)
        logging.info(pformat(args.__dict__))

        try:
            logging.info(pformat(get_git_info()))
        except:
            logging.info('Getting git info fails.')
    

    def mitigation(self):
        fix_random(self.args.random_seed)

        model = generate_cls_model(self.args.model,self.args.num_classes)

        normalization = get_dataset_normalization(args.dataset)
        denormalization = get_dataset_denormalization(normalization)

        def poison_gen_infer(x):
            x = x.clone()
            # input x is a batch of images with normalization
            x = denormalization(x)

            grid = torch.zeros_like(x)
            grid = grid + args.pix_value

            mask = torch.zeros_like(x)

            if args.trigger_type == 0:
                if args.dataset=='tiny':
                    mask[:,:,:,0:5]=1
                    mask[:,:,:,-5:]=1
                    mask[:,:,0:5,:]=1
                    mask[:,:,-5:,:]=1
                else:
                    mask[:,:,:,0:1]=1
                    mask[:,:,:,-1:]=1
                    mask[:,:,0:1,:]=1
                    mask[:,:,-1:,:]=1
                    

            elif args.trigger_type == 1:
                mask[:,:,0:7,0:7]=1

            elif args.trigger_type == 2:
                mask[:,:,::2,0]=1
                mask[:,:,::2,-1]=1
                mask[:,:,0,::2]=1
                mask[:,:,-1,::2]=1

            else:
                raise ValueError(f'Invalid trigger_type {args.trigger_type}')

            x = x * (1-mask) + grid * mask
            
            x = normalization(x)
            return x


        def poison_gen(x):
            x = x.clone()
            # input x is a single image in range 0-1 without normalization

            grid = torch.zeros_like(x)
            grid = grid + args.pix_value

            mask = torch.zeros_like(x)

            if args.trigger_type == 0:
                if args.dataset=='tiny':
                    mask[:,:,0:5]=1
                    mask[:,:,-5:]=1
                    mask[:,0:5,:]=1
                    mask[:,-5:,:]=1
                else:
                    mask[:,:,0:1]=1
                    mask[:,:,-1:]=1
                    mask[:,0:1,:]=1
                    mask[:,-1:,:]=1
                    

            elif args.trigger_type == 1:
                mask[:,0:7,0:7]=1

            elif args.trigger_type == 2:
                mask[:,::2,0]=1
                mask[:,::2,-1]=1
                mask[:,0,::2]=1
                mask[:,-1,::2]=1

            else:
                raise ValueError(f'Invalid trigger_type {args.trigger_type}')

            x = x * (1-mask) + grid * mask            
            return x

        def target_gen(x):
            return (x+args.target_type )%self.args.num_classes
        
        model_bd = bd_warp(model, poison_gen_infer, target_gen, self.args)

        if "," in self.args.device:
            model = torch.nn.DataParallel(model, device_ids=[int(i) for i in self.args.device[5:].split(",")])
            self.args.device = f'cuda:{model.device_ids[0]}'
            model.to(self.args.device)
        else:
            model.to(self.args.device)
        
        # a. get some clean data
        logging.info("Fetch some samples from clean train dataset.")
        
        # only keep resize and to tensor
        self.result['bd_train'].wrap_img_transform = transforms.Compose([transforms.Resize((args.input_height, args.input_width)), transforms.ToTensor()])
        self.result['clean_train'].wrap_img_transform = transforms.Compose([transforms.Resize((args.input_height, args.input_width)), transforms.ToTensor()])
        
        # fetch all datas        
        x_samples = [x for x, y, *other_info in self.result["bd_train"]]
        y_samples = [y for x, y, *other_info in self.result["bd_train"]]
        indicators = [other_info[1] for x, y, *other_info in self.result["bd_train"]]

        x_clean_samples = [x for x, y, *other_info in self.result["clean_train"]]
        y_clean_samples = [y for x, y, *other_info in self.result["clean_train"]]
        
        # select
        num_samples_class = int(len(x_samples) * self.args.ratio/self.args.num_classes)
        
        # for each class, choose num_samples_class samples
        selected_index = []
        for i in range(self.args.num_classes):
            idx_i = [idx for idx, y in enumerate(y_samples) if y == i]
            selected_index += random.sample(idx_i, num_samples_class)
            
        logging.info(f"select {num_samples_class} samples for each class.")

        # construct the defensive dataset
        x_defensive = [poison_gen(x_clean_samples[idx]) for idx in selected_index]
        y_defensive = [target_gen(y_clean_samples[idx]) for idx in selected_index]
        indicators_defensive = [2 for _ in y_defensive]
        
        transforms_list = []
        transforms_list.append(transforms.RandomCrop((args.input_height, args.input_width), padding=4))
        transforms_list.append(transforms.RandomHorizontalFlip())
        transforms_list.append(get_dataset_normalization(args.dataset))
        trans =  transforms.Compose(transforms_list)

        transforms_list = []
        # add random scale of image
        transforms_list.append(transforms.RandomApply([lambda x: x * (0.5 + 0.5 * torch.rand(1,device = x.device))], p=0.2))
        transforms_list.append(transforms.RandomApply([lambda x: x + args.aug * torch.randn_like(x)], p=args.lmd_3/(args.lmd_2+args.lmd_3)))
        transforms_list.append(transforms.RandomApply([transforms.RandomRotation(10)], p=0.5))
        transforms_list.append(transforms.RandomCrop((args.input_height, args.input_width), padding=4))
        transforms_list.append(transforms.RandomHorizontalFlip(p=0.5))
        transforms_list.append(get_dataset_normalization(args.dataset))
        trans_defensive =  transforms.Compose(transforms_list)


        data_set_o = defensive_iter(x_samples, y_samples, indicators, trans, x_defensive, y_defensive, indicators_defensive, trans_defensive, sample_frequency = 5)
        
        logging.info(f'Construct defensive dataset with {len(data_set_o)} samples.')
        
        data_loader = torch.utils.data.DataLoader(data_set_o, batch_size=self.args.batch_size, num_workers=self.args.num_workers, shuffle=True, pin_memory=args.pin_memory, drop_last=True)
        trainloader = data_loader
        
        ## set testing dataset
        test_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = False)
        data_bd_testset = self.result['bd_test']
        data_bd_testset.wrap_img_transform = test_tran
        data_bd_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,drop_last=False, shuffle=True,pin_memory=args.pin_memory)

        data_clean_testset = self.result['clean_test']
        data_clean_testset.wrap_img_transform = test_tran
        data_clean_loader = torch.utils.data.DataLoader(data_clean_testset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,drop_last=False, shuffle=True,pin_memory=args.pin_memory)

        clean_test_loss_list = []
        bd_test_loss_list = []
        test_acc_list = []
        test_asr_list = []
        test_bd_acc_list = []
        test_bd_asr_list = []

        # b. unlearn the backdoor model by the pertubation
        logging.info("=> Conducting Defence..")
        model.eval()

        clean_test_loss_avg_over_batch, \
                    bd_test_loss_avg_over_batch, \
                    _, \
                    test_acc, \
                    test_asr, \
                    _ = self.eval_step(
                        model,
                        data_clean_loader,
                        data_bd_loader,
                        args,
                    )        

        agg = Metric_Aggregator()
        outer_opt, scheduler = argparser_opt_scheduler(model, self.args)
            
        for round in range(args.epochs):
            model.train()
            batch_loss_list = []
            for images, labels, indicator in trainloader:
                images = images.to(args.device)
                labels = labels.to(args.device)
                indicator = indicator.to(args.device)
                wight = torch.ones_like(labels).view(-1,1)

                logits =  model(images)
                logits = logits * wight

                loss = F.cross_entropy(logits, labels)

                print(f'loss: {loss.item()}, malicious {torch.sum(indicator==1).item()}, cleans {torch.sum(indicator==0).item()}, defensive {torch.sum(indicator==2).item()}')
                outer_opt.zero_grad()
                loss.backward()
                outer_opt.step()
            scheduler.step()

            # eval bd mode
            model.eval()
            print(f'Clean  Mode: Test without defensive')            
            clean_test_loss_avg_over_batch, \
            bd_test_loss_avg_over_batch, \
            ra_test_loss_avg_over_batch, \
            test_acc, \
            test_asr, \
            test_ra = self.eval_step(
                model,
                data_clean_loader,
                data_bd_loader,
                args,
            )


            
            print(f'BD  Mode: Test with defensive')            
            bd_clean_test_loss_avg_over_batch, \
            bd_bd_test_loss_avg_over_batch, \
            bd_ra_test_loss_avg_over_batch, \
            test_bd_acc, \
            test_bd_asr, \
            bd_test_ra = self.eval_step(
                model_bd,
                data_clean_loader,
                data_bd_loader,
                args,
            )

            agg({
                "epoch": round,

                # match with default metric in BackdoorBench
                "clean_test_loss_avg_over_batch": bd_clean_test_loss_avg_over_batch,
                "bd_test_loss_avg_over_batch": bd_bd_test_loss_avg_over_batch,
                "ra_test_loss_avg_over_batch": bd_ra_test_loss_avg_over_batch,
                "test_acc": test_bd_acc,
                "test_asr": test_bd_asr,
                "test_ra": bd_test_ra,
                
                # without defensive
                "clean_test_loss_avg_over_batch_wo_defensive": clean_test_loss_avg_over_batch,
                "bd_test_loss_avg_over_batch_wo_defensive": bd_test_loss_avg_over_batch,
                "bd_ra_test_loss_avg_over_batch_wo_defensive": ra_test_loss_avg_over_batch,
                "test_acc_wo_defensive": test_acc,
                "test_asr_wo_defensive": test_asr,
                "test_ra_wo_defensive": test_ra,

                # with defensive
                "clean_test_loss_avg_over_batch_w_defensive": bd_clean_test_loss_avg_over_batch,
                "bd_test_loss_avg_over_batch_w_defensive": bd_bd_test_loss_avg_over_batch,
                "ra_test_loss_avg_over_batch_w_defensive": bd_ra_test_loss_avg_over_batch,
                "test_acc_w_defensive": test_bd_acc,
                "test_asr_w_defensive": test_bd_asr,
                "test_ra_w_defensive": bd_test_ra,
            })


            clean_test_loss_list.append(clean_test_loss_avg_over_batch)
            bd_test_loss_list.append(bd_test_loss_avg_over_batch)
            test_acc_list.append(test_acc)
            test_asr_list.append(test_asr)
            test_bd_acc_list.append(test_bd_acc)
            test_bd_asr_list.append(test_bd_asr)
            

            general_plot_for_epoch(
                {
                    "Test C-Acc": test_acc_list,
                    "Test ASR": test_asr_list,
                },
                save_path=f"{args.save_path}pdb_acc_like_metric_plots.png",
                ylabel="percentage",
            )

            general_plot_for_epoch(
                {
                    "Test ACC w/o Defensive": test_acc_list,
                    "Test ASR w/o Defensive": test_asr_list,
                    "Test C-Acc w/ Defensive": test_bd_acc_list,
                    "Test ASR  w/ Defensive": test_bd_asr_list,
                },
                save_path=f"{args.save_path}pdb_acc_full_like_metric_plots.png",
                ylabel="percentage",
            )


            general_plot_for_epoch(
                {
                    "Test C-Acc": test_bd_acc_list,
                    "Test ASR": test_bd_asr_list,
                },
                save_path=f"{args.save_path}pdb_bd_acc_like_metric_plots.png",
                ylabel="percentage",
            )

            general_plot_for_epoch(
                {
                    "Test Clean Loss": clean_test_loss_list,
                    "Test Backdoor Loss": bd_test_loss_list,
                },
                save_path=f"{args.save_path}pdb_loss_metric_plots.png",
                ylabel="percentage",
            )

            agg.to_dataframe().to_csv(f"{args.save_path}pdb_df.csv")
        agg.summary().to_csv(f"{args.save_path}pdb_df_summary.csv")

        result = {}
        result['model'] = model
        save_defense_result(
            model_name=args.model,
            num_classes=args.num_classes,
            model=model.cpu().state_dict(),
            save_path=args.save_path,
        )
        return result

    def eval_step(
            self,
            netC,
            clean_test_dataloader,
            bd_test_dataloader,
            args,
    ):
        clean_metrics, clean_epoch_predict_list, clean_epoch_label_list = given_dataloader_test(
            netC,
            clean_test_dataloader,
            criterion=torch.nn.CrossEntropyLoss(),
            non_blocking=args.non_blocking,
            device=self.args.device,
            verbose=0,
        )
        clean_test_loss_avg_over_batch = clean_metrics['test_loss_avg_over_batch']
        test_acc = clean_metrics['test_acc']
        bd_metrics, bd_epoch_predict_list, bd_epoch_label_list = given_dataloader_test(
            netC,
            bd_test_dataloader,
            criterion=torch.nn.CrossEntropyLoss(),
            non_blocking=args.non_blocking,
            device=self.args.device,
            verbose=0,
        )
        bd_test_loss_avg_over_batch = bd_metrics['test_loss_avg_over_batch']
        test_asr = bd_metrics['test_acc']

        bd_test_dataloader.dataset.wrapped_dataset.getitem_all_switch = True  # change to return the original label instead
        ra_metrics, ra_epoch_predict_list, ra_epoch_label_list = given_dataloader_test(
            netC,
            bd_test_dataloader,
            criterion=torch.nn.CrossEntropyLoss(),
            non_blocking=args.non_blocking,
            device=self.args.device,
            verbose=0,
        )
        ra_test_loss_avg_over_batch = ra_metrics['test_loss_avg_over_batch']
        test_ra = ra_metrics['test_acc']
        bd_test_dataloader.dataset.wrapped_dataset.getitem_all_switch = False  # switch back

        return clean_test_loss_avg_over_batch, \
                bd_test_loss_avg_over_batch, \
                ra_test_loss_avg_over_batch, \
                test_acc, \
                test_asr, \
                test_ra


    def defense(self,result_file):
        self.set_result(result_file)
        self.set_logger()
        result = self.mitigation()
        return result


    def eval_attack(self, netC, net_ref, clean_test_dataloader, pert, args = None):  
        total_success = 0
        total_success_ref = 0
        total_success_common = 0
        total_success_shared = 0
        
        total_samples = 0
        for images, labels, *other_info in clean_test_dataloader:
            images = images.to(self.args.device)
            labels = labels.to(self.args.device)
            pert_image = self.get_perturbed_image(images=images, pert=pert)
            outputs = netC(pert_image)
            outputs_ref = net_ref(pert_image)
            _, predicted = torch.max(outputs.data, 1)
            _, predicted_ref = torch.max(outputs_ref.data, 1)
            total_success += (predicted != labels).sum().item()
            total_success_ref += (predicted_ref != labels).sum().item()
            total_success_common += (torch.logical_and(predicted != labels, predicted_ref != labels)).sum().item()
            total_success_shared += (torch.logical_and(predicted != labels, predicted_ref == predicted)).sum().item()
            total_samples += labels.size(0)
        
        return total_success/total_samples, total_success_ref/total_samples, total_success_common/total_samples, total_success_shared/total_samples
    
    def eval_binary(self, netC, net_ref, bd_test_dataloader, pert, args = None):  
        total_success = 0
        total_success_ref = 0
        total_success_common = 0
        total_success_shared = 0
        
        total_samples = 0
        for images, labels, *other_info in bd_test_dataloader:
            images = images.to(self.args.device)
            labels = labels.to(self.args.device)
            pert_image = self.get_perturbed_image(images=images, pert=pert)
            outputs = netC(pert_image)
            outputs_ref = net_ref(pert_image)
            _, predicted = torch.max(outputs.data, 1)
            _, predicted_ref = torch.max(outputs_ref.data, 1)
            total_success += (predicted != labels).sum().item()
            total_success_ref += (predicted_ref != labels).sum().item()
            total_success_common += (torch.logical_and(predicted != labels, predicted_ref != labels)).sum().item()
            total_success_shared += (torch.logical_and(predicted != labels, predicted_ref == predicted)).sum().item()
            total_samples += labels.size(0)
        
        return total_success/total_samples, total_success_ref/total_samples, total_success_common/total_samples, total_success_shared/total_samples

    def defense(self,result_file):
        self.set_result(result_file)
        self.set_logger()
        result = self.mitigation()
        return result
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=sys.argv[0])
    pdb.add_arguments(parser)
    args = parser.parse_args()
    pdb_method = pdb(args)
    if "result_file" not in args.__dict__:
        args.result_file = 'defense_test_badnet'
    elif args.result_file is None:
        args.result_file = 'defense_test_badnet'
    result = pdb_method.defense(args.result_file)
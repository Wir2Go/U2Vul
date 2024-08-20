import os, sys
import json
import time
import datetime
import multiprocessing as mp

import math
import torch 
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model import GNN as encoder
from model import DINO as model
from model_utils import DINO as utils
from multiprocessing import shared_memory

torch.multiprocessing.set_sharing_strategy('file_system')

LR = 5e-4
batch_size = 1024
num_workers = 16
epochs = 100
warmup_epochs = 10
save_epoch = 5
device = "cuda:0"

ncrops = 8

seed = 42
# num_sample = 1158283


word_num = 8000
max_len = 512
emb_dim = 512
out_dim = 1024

parent_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
data_path = os.path.join(parent_path, 'data/CPG_Emb_5_tensor')
model_path = os.path.join(parent_path, 'model/pretrained_ggnn')
output_path = os.path.join(parent_path, 'log/pretrain_ggnn_dino')
os.makedirs(model_path, exist_ok=True)
os.makedirs(output_path, exist_ok=True)

class Bunch:
    def __init__(self, **entries):
        self.__dict__.update(entries)

custom_args = {

    'lr': 5e-4,
    'min_lr': 1e-6,
    'batch_size': batch_size,
    'out_dim': out_dim,
    'local_crops_number': 8,

    'warmup_epochs': 10,
    'epochs': 100,

    'momentum_teacher': 0.996,
    'warmup_teacher_temp': 0.04,
    'teacher_temp': 0.04,
    'warmup_teacher_temp_epochs': 0,


    'weight_decay': 0.04,
    'weight_decay_end': 0.04,
    'clip_grad': 3.0,

    'optimizer': 'adamw',
    'drop_path_rate': 0.1,

    'norm_last_layer': True,
    'freeze_last_layer': 1,

    'arch': 'GGNN',
    'use_fp16': True,
    'device': 'cuda:0',

    'saveckp_freq': 5,
    'output_dir': output_path
}

args = Bunch(**custom_args)



def train_dino(args):
    # utils.init_distributed_mode(args)
    utils.fix_random_seeds(seed)
    cudnn.benchmark = True

    # ============ preparing data ... ============
    fl = shared_memory.ShareableList(os.listdir(data_path))
    dataset = utils.DINO_CPG_Dataset(raw_dir=data_path, fl=fl, ncrops=args.local_crops_number)
    # sampler = RandomSampler(dataset, replacement=True)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=utils.dino_collate,
        shuffle=True,
        drop_last=True
    )

    # ============ building student and teacher networks ... ============
    student = encoder.CPGNN(in_size=emb_dim, out_size=emb_dim)
    mlp_student = model.DiNOHead(in_dim=emb_dim, out_dim=out_dim)
    teacher = encoder.CPGNN(in_size=emb_dim, out_size=emb_dim)
    mlp_teacher = model.DiNOHead(in_dim=emb_dim, out_dim=out_dim)
    student = model.DiNOWrapper(student, mlp_student)
    teacher = model.DiNOWrapper(teacher, mlp_teacher)
    teacher.load_state_dict(student.state_dict())
    # teacher = copy.deepcopy(student)
    teacher.mlp.last_layer.weight_g.requires_grad = custom_args["norm_last_layer"]

    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)


    # multi-crop wrapper handles forward with inputs of different resolutions
    train_model = model.DiNO(student=student, teacher=teacher, device=args.device).to(args.device)

    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============
    dino_loss = model.DINOLoss(
        args.out_dim,
        2 + args.local_crops_number,  # total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).to(args.device)

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting DINO training !")
    for epoch in range(start_epoch, args.epochs):

        # ============ training one epoch of DINO ... ============
        # train_stats = train_one_epoch(student, teacher, dino_loss,
        #     data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
        #     epoch, fp16_scaler, args)
        train_stats = train_one_epoch(train_model, dino_loss,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, args)

        # ============ writing logs ... ============
        save_dict = {
            'student': train_model.student.state_dict(),
            'teacher': train_model.teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'dino_loss': dino_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and (epoch+1) % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch+1:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch+1}
        if utils.is_main_process():
            with open(f"{args.output_dir}/log.txt", "a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    return 0

# def train_one_epoch(train_model:model.DiNO, dino_loss, data_loader,
#                     optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
#                     fp16_scaler, args):
#         metric_logger = utils.MetricLogger(delimiter="  ")
#         header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
#         for (it, images) in enumerate(metric_logger.log_every(data_loader, 10, header)):
#             print(f'get {(len(data_loader) * epoch + it) * batch_size} samples\n', flush=True)


def train_one_epoch(train_model:model.DiNO, dino_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    # for it, (images, _) in enumerate(metric_logger.log_every(data_loader, 10, header)):
    for (it, images) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_output, student_output = train_model(images)  # only the 2 global views pass through the teacher
            loss = dino_loss(student_output, teacher_output, epoch)


        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(train_model.student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, train_model.student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(train_model.student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, train_model.student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        m = momentum_schedule[it]
        train_model.ema_update(m)

        # logging
        # torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

if __name__ == "__main__":
    mp.set_start_method('spawn')
    train_dino(args)

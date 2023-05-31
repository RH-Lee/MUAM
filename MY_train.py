import datetime
import os
import time
import torch.nn.functional as F
import importlib
from utils.utils import adaptive_pixel_intensity_loss
from .utils.metrics import Evaluation_metrics
from configs import cfg
from torch.optim import lr_scheduler
from utils.prepare_data_edge import SalData, pSalData, val_collate
from utils.utils import load_pretrained
import torch
import argparse
import albumentations as albu
from albumentations.pytorch.transforms import ToTensorV2
import torch.nn as nn


parser = argparse.ArgumentParser(description='PyTorch SOD FOR Tracer_res2net')

parser.add_argument(
    "--config",
    default="configs/MY_train.yml",
    metavar="FILE",
    help="path to config file",
    type=str,
)
args = parser.parse_args()
assert os.path.isfile(args.config)
cfg.merge_from_file(args.config)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.GPU)
if cfg.TASK == '':
    cfg.TASK = cfg.MODEL.ARCH

timenow = datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
check_point_dir = os.path.join(cfg.DATA.SAVEDIR, cfg.TASK, 'checkpoint')
logname = 'logtrain_' + cfg.TASK + "_" + timenow + '.txt'
val_logname = 'logval_' + cfg.TASK + "_" + timenow + '.txt'
logging_dir = os.path.join(cfg.DATA.SAVEDIR, cfg.TASK)
if not os.path.isdir(logging_dir):
    os.makedirs(logging_dir, exist_ok=True)
if not os.path.isdir(check_point_dir):
    os.mkdir(check_point_dir)
LOG_FOUT = open(os.path.join(logging_dir, logname), 'w')
VAL_LOG_FOUT = open(os.path.join(logging_dir, val_logname), 'w')


def log_string(out_str, display=True):
    out_str = str(out_str)
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    if display:
        print(out_str)


def val_log_string(out_str, display=False):
    out_str = str(out_str)
    VAL_LOG_FOUT.write(out_str + '\n')
    VAL_LOG_FOUT.flush()
    if display:
        print(out_str)


log_string(cfg)

best_mae = 1000000
best_epoch = -1
best_fossil_mae = 1000000
best_fossil_epoch = -1

def main():
    global cfg, best_mae, best_epoch, best_fossil_epoch, best_fossil_mae

    model_lib = importlib.import_module("model." + cfg.MODEL.ARCH)
    if cfg.AUTO.ENABLE:
        model = model_lib.build_model()
    else:
        print("Enable AUTO to train Tracer!")

    model.cuda()
    if cfg.SOLVER.METHOD == 'Adam_dynamic_weight_decay':
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-4)
    else:
        optimizer = None
        print("WARNING: Method not implmented.")
    if cfg.DATA.PRETRAIN != '':
        model = load_pretrained(model, cfg.DATA.PRETRAIN)
    start_epoch = 0
    if cfg.DATA.RESUME != '':
        if os.path.isfile(cfg.DATA.RESUME):
            log_string("=> loading checkpoint '{}'".format(cfg.DATA.RESUME))
            checkpoint = torch.load(cfg.DATA.RESUME)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            log_string("=> loaded checkpoint '{}' (epoch {})".format(
                cfg.DATA.RESUME, checkpoint['epoch']))
        else:
            log_string("=> no checkpoint found at '{}'".format(
                cfg.DATA.RESUME))

    train_loader = prepare_data(cfg.DATA.DIR)

    if cfg.SOLVER.ADJUST_STEP:
        if cfg.SOLVER.LR_SCHEDULER == 'step':
            scheduler = lr_scheduler.StepLR(optimizer,
                                                 step_size=10,
                                                 gamma=0.9)
        else:
            raise ValueError("Unsupported scheduler.")

    for epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCHS):
        if (cfg.SOLVER.FINETUNE.ADJUST_STEP
            and epoch > cfg.AUTO.FINETUNE) or cfg.SOLVER.ADJUST_STEP:

            lr = scheduler.get_last_lr()[0]
            log_string("lr: " + str(lr))

        train(train_loader, model, optimizer, epoch)
        scheduler.step()


        val_loader = prepare_val_data(cfg.VAL.DIR)
        mae, max_f, avg_f, s_score = val(val_loader, model, epoch)

        is_best = mae < best_mae
        best_mae = min(mae, best_mae)
        if is_best:
            best_epoch = epoch + 1
            val_loader = prepare_val_data("datasets/fossil")
            fo_mae, fo_max_f, fo_avg_f, fo_s_score = val(val_loader, model, epoch)
            val_log_string(" fo_mae: " + str(fo_mae) +
                           " fo_max_f: " + str(fo_max_f) + " fo_avg_f: " + str(fo_avg_f) +
                           " fo_s_score: " + str(fo_s_score))
        log_string(" epoch: " + str(epoch + 1) + " mae: " + str(mae) +
                   " best_epoch: " + str(best_epoch) + " best_mae: " +
                   str(best_mae))
        val_log_string(" epoch: " + str(epoch + 1) + " mae: " + str(mae) +
                       " max_f: " + str(max_f) + " avg_f: " + str(avg_f) +
                       " s_score: " + str(s_score) +
                       " best_epoch: " + str(best_epoch) + " best_mae: " +
                       str(best_mae))
        # Save checkpoint
        save_file = os.path.join(
            check_point_dir, 'checkpoint_epoch{}.pth.tar'.format(epoch + 1))
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'arch': cfg.MODEL.ARCH,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            },
            filename=save_file)



def train(train_loader, model, optimizer, epoch):
    accumulation_steps = 8
    log_string('Memory useage: %.4fM' %
               (torch.cuda.max_memory_allocated() / 1024.0 / 1024.0))
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # switch to train mode
    model.train()
    end = time.time()
    for i, data in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)
        # prepare input
        input = data['img'].float()
        target = data['gt'].float()
        edge = data['edge'].float()
        input = torch.autograd.Variable(input).cuda()
        edge = edge.cuda()
        target = target.cuda()


        outputs, edge_mask, ds_map = model(input)

        loss = 0

        loss += adaptive_pixel_intensity_loss(edge_mask, edge)
        loss += adaptive_pixel_intensity_loss(outputs, target)
        loss += adaptive_pixel_intensity_loss(ds_map[0], target)
        loss += adaptive_pixel_intensity_loss(ds_map[1], target)
        loss += adaptive_pixel_intensity_loss(ds_map[2], target)
        loss = loss / accumulation_steps
        # measure accuracy and record loss without flops
        losses.update(loss.item(), input.size(0))

        loss.backward()


        if ((i + 1) % accumulation_steps) == 0:
            nn.utils.clip_grad_norm_(model.parameters(), 2)
            optimizer.step()
            optimizer.zero_grad()
        batch_time.update(time.time() - end)
        end = time.time()
        if i % cfg.PRINT_FREQ == 0:
            if cfg.AUTO.FLOPS.ENABLE and epoch < cfg.AUTO.FINETUNE:
                log_string('Epoch: [{0}][{1}/{2}]\t'
                           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                           'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    epoch + 1,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses))
            else:
                log_string('Epoch: [{0}][{1}/{2}]\t'
                           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                           'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    epoch+1,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses))


def val(val_loader, model, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    maes = AverageMeter()
    # maes_0 = AverageMeter()
    maxf = AverageMeter()
    avgf = AverageMeter()
    s_m = AverageMeter()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Eval_tool = Evaluation_metrics('DUTS', device)
    # switch to eval mode


    model.eval()
    end = time.time()
    with torch.no_grad():
        for i, [img, target, h, w] in enumerate(val_loader):

            # measure data loading time
            data_time.update(time.time() - end)
            # prepare input
            input = img.float().cuda()

            output = model(input)[0]
            for idx in range(input.size(0)):
                output_resize = (F.interpolate(
                    output[idx].unsqueeze(dim=0),
                    size=(h[idx], w[idx]),
                    mode='bilinear') * 255.0).int().float() / 255.0
                this_target = target[idx].float().cuda().unsqueeze(dim=0)
                # Metric
                mae, max_f, avg_f, s_score = Eval_tool.cal_total_metrics(output_resize, this_target)
                maes.update(mae, 1)
                maxf.update(max_f, 1)
                avgf.update(avg_f, 1)
                s_m.update(s_score, 1)
            batch_time.update(time.time() - end)
            end = time.time()
            if i % cfg.VAL.PRINT_FREQ == 0:
                print('ValEpoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'MAE {mae.val:.4f} ({mae.avg:.4f})\t'
                      'MAX-F {max_f.val:.4f} ({max_f.avg:.4f})\t'
                      'AVG-F {avg_f.val:.4f} ({avg_f.avg:.4f})\t'
                      'S-Score {s_score.val:.4f} ({s_score.avg:.4f})\t'.format(
                    epoch + 1,
                    i,
                    len(val_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    mae=maes,
                    max_f=maxf,
                    avg_f=avgf,
                    s_score=s_m))
    return maes.avg, maxf.avg, avgf.avg, s_m.avg


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def prepare_data(data_dir):
    transforms = albu.Compose([
        albu.OneOf([
            albu.HorizontalFlip(),
            albu.VerticalFlip(),
            albu.RandomRotate90()
        ], p=0.5),
        albu.OneOf([
            # albu.RandomContrast(),
            albu.RandomBrightnessContrast(),
            albu.RandomGamma(),
            albu.RandomBrightness(),
        ], p=0.5),
        albu.OneOf([
            albu.MotionBlur(blur_limit=5),
            albu.MedianBlur(blur_limit=5),
            albu.GaussianBlur(blur_limit=5),
            albu.GaussNoise(var_limit=(5.0, 20.0)),
        ], p=0.5),
        albu.Resize(cfg.DATA.IMAGE_H, cfg.DATA.IMAGE_W, always_apply=True),
        albu.Normalize([0.485, 0.456, 0.406],
                       [0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    dataset = SalData(data_dir, (cfg.DATA.IMAGE_H, cfg.DATA.IMAGE_W), transform=transforms)
    train_loader = torch.utils.data.DataLoader(dataset,
                                               shuffle=True,
                                               batch_size=cfg.DATA.BATCH_SIZE,
                                               num_workers=cfg.DATA.WORKERS,
                                               drop_last=True)
    return train_loader


def prepare_val_data(val_data_dir):
    # prepare dataloader for val
    transforms = albu.Compose([
        albu.Resize(cfg.DATA.IMAGE_H, cfg.DATA.IMAGE_W, always_apply=True),
        albu.Normalize([0.485, 0.456, 0.406],
                       [0.229, 0.224, 0.225]),
        # ToTensorV2(),
    ])
    val_dataset = pSalData(val_data_dir, (cfg.DATA.IMAGE_H, cfg.DATA.IMAGE_W),
                          transform=transforms)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             shuffle=False,
                                             batch_size=cfg.DATA.BATCH_SIZE,
                                             collate_fn=val_collate,
                                             num_workers=cfg.DATA.WORKERS,
                                             drop_last=False)
    return val_loader


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum = self.sum + val * n
        self.count = self.count + n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()

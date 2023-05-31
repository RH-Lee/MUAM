
import torch.nn.functional as F
import numpy as np
import os
import argparse
from scipy import misc  # NOTES: pip install scipy == 1.2.2 (prerequisite!)


from .utils.metrics import Evaluation_metrics
import albumentations as albu
import argparse
import importlib
import math
import os
from utils.prepare_data_edge import pSalData, val_collate
import numpy as np
import skimage
import torch
from configs import cfg
from skimage import io
from skimage.transform import resize


parser = argparse.ArgumentParser(description='PyTorch SOD')

parser.add_argument(
    "--config",
    default="configs/MY.yml",
    metavar="FILE",
    help="path to config file",
    type=str,
)
args = parser.parse_args()
assert os.path.isfile(args.config)
cfg.merge_from_file(args.config)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.GPU)
if cfg.TASK == '':
    cfg.TASK = cfg.MODEL.ARCH

print(cfg)
best_mae = 1000000
best_epoch = -1
import datetime
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

def main():
    # model = CSSINet().cuda()
    global cfg, best_mae, best_epoch
    model_lib = importlib.import_module("model." + cfg.MODEL.ARCH)
    # predefine_file = cfg.TEST.MODEL_CONFIG
    model = model_lib.build_model()
    model.cuda()

    this_checkpoint = cfg.TEST.CHECKPOINT
    if os.path.isfile(this_checkpoint):
        print("=> loading checkpoint '{}'".format(this_checkpoint))
        checkpoint = torch.load(this_checkpoint)
        loadepoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(
            this_checkpoint, checkpoint['epoch']))

        val_loader = prepare_val_data(cfg.VAL.DIR)
        mae, max_f, avg_f, s_score = val(val_loader, model)
        print(f'| MAX_F:{max_f:.8f} | MAE:{mae:.8f} '
              f'| S_Measure:{s_score:.8f} | AVG_F:{avg_f:.8f} ')

        # test(model, cfg.TEST.DATASETS, loadepoch)

        # eval(cfg.TASK, loadepoch)

    else:
        print(this_checkpoint, "Not found.")



def test(model, test_datasets, epoch):
    model.eval()
    print("Start testing.")
    for dataset in test_datasets:
        sal_save_dir = os.path.join(cfg.DATA.SAVEDIR, cfg.TASK,
                                    dataset + '_' + str(epoch))
        os.makedirs(sal_save_dir, exist_ok=True)
        img_dir = os.path.join(cfg.TEST.DATASET_PATH, dataset, 'images')
        img_list = os.listdir(img_dir)
        count = 0
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        with torch.no_grad():
            for img_name in img_list:
                img = skimage.img_as_float(
                    io.imread(os.path.join(img_dir, img_name)))
                h, w = img.shape[:2]
                if cfg.TEST.IMAGE_H != 0 and cfg.TEST.IMAGE_W != 0:
                    img = resize(img, (cfg.TEST.IMAGE_H, cfg.TEST.IMAGE_W),
                                 mode='reflect',
                                 anti_aliasing=False)
                else:
                    if h % 16 != 0 or w % 16 != 0:
                        img = resize(
                            img,
                            (math.ceil(h / 16) * 16, math.ceil(w / 16) * 16),
                            mode='reflect',
                            anti_aliasing=False)
                img = np.transpose((img - mean) / std, (2, 0, 1))
                img = torch.unsqueeze(torch.FloatTensor(img), 0)
                input_var = torch.autograd.Variable(img)
                input_var = input_var.cuda()
                predict = model(input_var)
                predict = predict[0]
                # predict = torch.sigmoid(predict.squeeze(0).squeeze(0))
                predict = predict.squeeze(0).squeeze(0)
                predict = predict.data.cpu().numpy()
                predict = (resize(
                    predict, (h, w), mode='reflect', anti_aliasing=False) *
                           255).astype(np.uint8)
                save_file = os.path.join(sal_save_dir, img_name[0:-4] + '.png')
                io.imsave(save_file, predict)
                count += 1
        print('Dataset: {}, {} images'.format(dataset, len(img_list)))

def val(val_loader, model):
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
    with torch.no_grad():
        for i, [img, target, h, w] in enumerate(val_loader):

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

            print('MAE {mae.val:.4f} ({mae.avg:.4f})\t'
                  'MAX-F {max_f.val:.4f} ({max_f.avg:.4f})\t'
                  'AVG-F {avg_f.val:.4f} ({avg_f.avg:.4f})\t'
                  'S-Score {s_score.val:.4f} ({s_score.avg:.4f})\t'.format(
                mae=maes,
                max_f=maxf,
                avg_f=avgf,
                s_score=s_m))
    return maes.avg, maxf.avg, avgf.avg, s_m.avg


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

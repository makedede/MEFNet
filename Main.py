import argparse
import TrainModel
import os
import ast

#Training Code

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=ast.literal_eval, default=None)
    parser.add_argument("--use_cuda", type=ast.literal_eval, default=None)
    parser.add_argument("--seed", type=int, default=2019)

    parser.add_argument("--trainset", type=str, default="./images/")
    parser.add_argument("--finetuneset", type=str, default="./images/")
    parser.add_argument("--testset", type=str, default="./images/")

    parser.add_argument('--ckpt_path', default='./checkpoint/', type=str,
                        metavar='PATH', help='path to checkpoints')
    parser.add_argument('--ckpt', default=None, type=str, help='name of the checkpoint to load')

    parser.add_argument('--fused_img_path', default='./fused_result/', type=str,
                        metavar='PATH', help='path to save images')
    parser.add_argument('--weight_map_path', default='./weighting_maps/', type=str,
                        metavar='PATH', help='path to save weight maps')

    parser.add_argument("--low_size", type=int, default=128)
    parser.add_argument("--high_size", type=int, default=512, help='None means random resolution')
    parser.add_argument("--max_epochs", type=int, default=4)
    parser.add_argument("--finetune_epochs", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--decay_interval", type=int, default=1000)
    parser.add_argument("--decay_ratio", type=float, default=0.1)
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--epochs_per_eval", type=int, default=10)#
    parser.add_argument("--epochs_per_save", type=int, default=100)#

    return parser.parse_args()


def main(cfg):
    t = TrainModel.Trainer(cfg)
    if cfg.train:
        t.fit()
    else:
        t.eval(0)


if __name__ == "__main__":
    config = parse_config()

    if not os.path.exists(config.ckpt_path):
        os.makedirs(config.ckpt_path)
    if not os.path.exists(config.fused_img_path):
        os.makedirs(config.fused_img_path)
    if not os.path.exists(config.weight_map_path):
        os.makedirs(config.weight_map_path)
    main(config)

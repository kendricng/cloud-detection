import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--train_dataset_dir', type=str, default=r'/Users/stas/workspaces/38-Cloud', help='Train dataset path')
parser.add_argument('--log_dir', type=str, default='./logs', help='Logging path')
parser.add_argument('--saved_model_dir', type=str, default='./weights/', help='Saved pb model path')

parser.add_argument('--n_epochs', type=int, default=60)
parser.add_argument('--validation_ratio', type=float, default=0.2)

parser.add_argument('--batch_size', type=int, default=12, help='Total batch size for all GPUs')
parser.add_argument('--multi_gpus', type=bool, default=False)
parser.add_argument('--init_learning_rate', type=float, default=3e-4)
parser.add_argument('--cosine_alpha', type=float, default=1e-2)
parser.add_argument('--cosine_epochs', type=int, default=45)
parser.add_argument('--image_size', type=int, default=192, help='Image target size')
parser.add_argument('--random_seed', type=int, default=42, help='Random seed')

args = parser.parse_args()
params = vars(args)

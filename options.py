import argparse
import os


class TrainOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--full_dose_path', required=True, help='Path to full dose images')
        self.parser.add_argument('--quarter_dose_path', required=True, help='Path to quarter dose images')
        self.parser.add_argument('--dataset_ratio', type=float, default=0.075, help='The ratio of dataset to use (in case of big dataset)')
        self.parser.add_argument('--train_ratio', type=float, default=0.8, help='Ratio of train dataset to all dataset')
        self.parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
        self.parser.add_argument('--in_ch', type=int, default=1, help='Number of input image channels')
        self.parser.add_argument('--out_ch', type=int, default=1, help='Number of output image channels')
        self.parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
        self.parser.add_argument('--max_epoch', type=int, default=100, help='Maximum number of epochs')
        self.parser.add_argument('--continue_to_train', action='store_true', help='Continue any interrupted training')
        self.parser.add_argument('--path_to_save', type=str, required=True, help='Path to save the trained model')
        self.parser.add_argument('--ckpt_path', type=str, default='-', help='Path to trained and saved checkpoint model')
        self.parser.add_argument('--validation_freq', type=int, default=10, help='Frequency to run validation')
        self.parser.add_argument('--save_freq', type=int, default=20, help='Frequency to save model')
        self.parser.add_argument('--batch_number', type=int, default=3, help='Number of a batch in validation to show the sample images')
        
        
        self.parser.add_argument('--n_layer', type=int, default=4, help='Number of transformer block layers')
        self.parser.add_argument('--num_blocks', nargs='+', type=int, default=[4, 6, 6, 8], help='Number of transformer blocks')
        self.parser.add_argument('--dim', type=int, default=48, help='Transformer block dimension')
        self.parser.add_argument('--num_refinement_blocks', type=int, default=2, help='Number of refinement blocks')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print(f'{k}: {v}')
        print('-------------- End ----------------')

        return self.opt
    
    
class TestOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--full_dose_path', required=True, help='Path to full dose images')
        self.parser.add_argument('--quarter_dose_path', required=True, help='Path to quarter dose images')
        self.parser.add_argument('--dataset_ratio', type=float, default=0.05, help='The ratio of dataset to use (in case of big dataset)')
        self.parser.add_argument('--train_ratio', type=float, default=0.8, help='Ratio of train dataset to all dataset')
        self.parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
        self.parser.add_argument('--in_ch', type=int, default=1, help='Number of input image channels')
        self.parser.add_argument('--out_ch', type=int, default=1, help='Number of output image channels')
        self.parser.add_argument('--ckpt_path', type=str, required = True, help='Path to trained and saved checkpoint model')
        self.parser.add_argument('--output_root', type=str, required = True, help='Path to save denoised imgs')
        
        self.parser.add_argument('--n_layer', type=int, default=4, help='Number of transformer block layers')
        self.parser.add_argument('--num_blocks', nargs='+', type=int, default=[4, 6, 6, 8], help='Number of transformer blocks')
        self.parser.add_argument('--dim', type=int, default=48, help='Transformer block dimension')
        self.parser.add_argument('--num_refinement_blocks', type=int, default=2, help='Number of refinement blocks')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print(f'{k}: {v}')
        print('-------------- End ----------------')

        return self.opt

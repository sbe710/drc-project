import subprocess
import sys
import argparse

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

parser = argparse.ArgumentParser()
parser.add_argument('--test_folder', default='./input', type=str, help='folder path to input images')
parser.add_argument('--cuda', default=False, type=str2bool, help='Use cuda for inference')
parser.add_argument('--trained_model', default='./weights/craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--image_folder', default='./Results/CropWords/', help='path to image_folder which contains cropped text images')
parser.add_argument('--saved_model', default='./pretrained_model/TPS-ResNet-BiLSTM-Attn.pth', help="path to saved_model to evaluation")
args = vars(parser.parse_args())

#print(args['test_folder'])

subprocess.call([sys.executable, "./detect.py", f"--test_folder={args['test_folder']}", f"--cuda={args['cuda']}", f"--trained_model={args['trained_model']}"])
subprocess.call([sys.executable, "./crop_images.py"])
subprocess.call([sys.executable, "./recog.py", f"--image_folder={args['image_folder']}", f"--saved_model={args['saved_model']}"])
#subprocess.call([sys.executable, "./class.py"])
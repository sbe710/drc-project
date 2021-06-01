import subprocess
import sys
import shutil
from os import path
import argparse

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

if path.exists('./Results'):
    shutil.rmtree('./Results')

parser = argparse.ArgumentParser()
parser.add_argument('--test_folder', default='./input', type=str, help='folder path to input images')
parser.add_argument('--cuda', default=False, type=str2bool, help='Use cuda for inference')
parser.add_argument('--trained_model', default='./weights/craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--image_folder', default='./Results/CropWords/', help='path to image_folder which contains cropped text images')
# parser.add_argument('--saved_model', default='./pretrained_model/TPS-ResNet-BiLSTM-Attn.pth', help="path to saved_model to evaluation")
# parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')#0123456789абвгдежзийклмнопрстуфхцчшщъыьэюя
parser.add_argument('--language', default='english', help="dataset language")
args = vars(parser.parse_args())

# print(args['test_folder'])

if args['language'] == "english":
    savedModel = './pretrained_model/TPS-ResNet-BiLSTM-Attn.pth'
    characterBank = '0123456789abcdefghijklmnopqrstuvwxyz'
else:
    savedModel = './pretrained_model/best_accuracy.pth'
    characterBank = '0123456789абвгдежзийклмнопрстуфхцчшщъыьэюя'

subprocess.call(['python3', "detect.py", f"--test_folder={args['test_folder']}", f"--cuda={args['cuda']}", f"--trained_model={args['trained_model']}"])
subprocess.call(['python3', "crop_images.py"])
subprocess.call(['python3', "recog.py", f"--image_folder={args['image_folder']}", f"--saved_model={savedModel}", f"--character={characterBank}"])
subprocess.call(['python3', "classification-predict.py", f"--language={args['language']}"])
#subprocess.call([sys.executable, "./class.py"])
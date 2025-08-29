from utils import *
from conv import ConvNet
import argparse
import os

def find_circle(img_data):
	return nn.serve_predictions(img_data)

def main_modified():
	data, labels = [], []
	for _ in range(1000):
		params, img = noisy_circle(200, 50, 2)
		data.append(img)
		labels.append(params)
	
	detected = find_circle(np.reshape(data, [-1, 200, 200, 1]))
	results = accuracy(detected, np.array(labels))
	print(results)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--saved_ckpt_path', type=str, default='/files/model-0/model.ckpt-25',
		help='Path to saved model checkpoint files.')
	parser.add_argument('--version', type=int, default=0,
		help='To test model-1 use --version=1 else use (default) --version=0')
	args = parser.parse_args()

	if not os.path.exists(args.saved_ckpt_path + '.meta'):
		raise Exception('Invalid checkpoint (prefix) path specified')
	
	nn = ConvNet(restore=True, version=args.version, saved_ckpt_path=args.saved_ckpt_path)
	main_modified()

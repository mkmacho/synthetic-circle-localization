If you have docker installed, simply enter the repo and run the `run.sh` script. 

Once inside the container you can train a convolutional network to localize "the circle" by calling:
	`python conv.py` 

You may also test model-0 by calling:
	`python main.py --saved_ckpt_path=/files/model-0/model.ckpt-5 --version=0`
or test model-1 (by default) simply calling:
	`python main.py`.



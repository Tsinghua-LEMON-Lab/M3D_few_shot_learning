## M3D_few_shot_learning

To reproduce our result, you need to train first, and then test. The training process is set on cosine similarity and the test process is set on hamming distance.
* Default mode is training mode. To switch to training: set `test_only=True` in mainOmniglot.py
* You may need to download Omniglot dataset from https://github.com/brendenlake/omniglot/ and place them under folder `datasets`.
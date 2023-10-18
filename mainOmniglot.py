
from datasets import omniglotNShot
from option import Options
from experiments.OneShotBuilder import OneShotBuilder
import tqdm
from logger import Logger
from utils import utility
import time

'''
:param batch_size: Experiment batch_size
:param classes_per_set: Integer indicating the number of classes per set
:param samples_per_class: Integer indicating samples per class
        e.g. For a 20-way, 1-shot learning task, use classes_per_set=20 and samples_per_class=1
             For a 5-way, 10-shot learning task, use classes_per_set=5 and samples_per_class=10
'''

# Experiment Setup
batch_size = 32
fce = False
classes_per_set = 5
samples_per_class = 1
channels = 1
# Training setup
total_epochs = 500
total_train_batches = 1000
total_val_batches = 100
total_test_batches = 250
k_bits = 128
print('current k_bit{}'.format(k_bits))
test_only = False
# Parse other options
args = Options().parse()

LOG_DIR = args.log_dir + '/1_run-batchSize_{}-fce_{}-classes_per_set{}-samples_per_class{}-channels{}-k_bits{}' \
    .format(batch_size,fce,classes_per_set,samples_per_class,channels,k_bits)

SAVE_DIR = args.log_dir + '/1_run-batchSize_{}-fce_{}-classes_per_set{}-samples_per_class{}-channels{}-k_bits{}-quant111' \
    .format(batch_size,fce,classes_per_set,samples_per_class,channels,k_bits)

print('save dir: {}'.format(LOG_DIR))

# create logger
# logger = Logger(LOG_DIR)

data = omniglotNShot.OmniglotNShotDataset(dataroot=args.dataroot, batch_size = batch_size,
                                          classes_per_set=classes_per_set,
                                          samples_per_class=samples_per_class)

obj_oneShotBuilder = OneShotBuilder(data,LOG_DIR, SAVE_DIR)
obj_oneShotBuilder.build_experiment(batch_size, k_bits, classes_per_set, samples_per_class, channels, fce, test_only)

if test_only:
    obj_oneShotBuilder.load_model()
    start = time.time()
    with tqdm.tqdm(total=total_epochs) as pbar_e:

        # for e in range(0, total_epochs):
        total_test_c_loss, total_test_accuracy = obj_oneShotBuilder.run_testing_epoch(
            total_test_batches=total_test_batches)
        print("Epoch {}: test_loss: {}, test_accuracy: {}".format(0, total_test_c_loss, total_test_accuracy))
    print("run time: {}".format(time.time() - start))
        # logger.log_value('test_loss', total_test_c_loss)
        # logger.log_value('test_acc', total_test_accuracy)
else:
    # obj_oneShotBuilder.load_model()
    logger = Logger(LOG_DIR)
    best_val = 0.
    with tqdm.tqdm(total=total_epochs) as pbar_e:
        for e in range(0, total_epochs):
            total_c_loss, total_accuracy = obj_oneShotBuilder.run_training_epoch(total_train_batches=total_train_batches)
            print("Epoch {}: train_loss: {}, train_accuracy: {}".format(e, total_c_loss, total_accuracy))

            total_val_c_loss, total_val_accuracy = obj_oneShotBuilder.run_validation_epoch(
                total_val_batches=total_val_batches)
            print("Epoch {}: val_loss: {}, val_accuracy: {}".format(e, total_val_c_loss, total_val_accuracy))

            logger.log_value('train_loss', total_c_loss)
            logger.log_value('train_acc', total_accuracy)
            logger.log_value('val_loss', total_val_c_loss)
            logger.log_value('val_acc', total_val_accuracy)

            if total_val_accuracy >= best_val:  # if new best val accuracy -> produce test statistics
                best_val = total_val_accuracy
                total_test_c_loss, total_test_accuracy = obj_oneShotBuilder.run_testing_epoch(
                    total_test_batches=total_test_batches)
                print("Epoch {}: test_loss: {}, test_accuracy: {}".format(e, total_test_c_loss, total_test_accuracy))
                logger.log_value('test_loss', total_test_c_loss)
                logger.log_value('test_acc', total_test_accuracy)
                obj_oneShotBuilder.save_model()
            else:
                total_test_c_loss = -1
                total_test_accuracy = -1

            pbar_e.update(1)
            logger.step()
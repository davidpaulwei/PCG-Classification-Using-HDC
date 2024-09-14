import hdc_functions as hdc
from hdc_functions import param
import read_mat
import time

# hyperparams.

# Hypervectors' dimensionality (D).
param.dim = 10000

# Features' value range. In "read_mat.py" the values are normalized into the range of [0, 1].
param.value_range = (0, 1)

# Number of classes (J). The goal of this task is to classify the samples into these classes.
param.num_classes = 5 

# Number of levels (M), or number of Level-HVs. Level-HVs are used to represent the values of features.
param.num_levels = 200

# Number of features (N). Features are represented by Feature-HVs.
param.num_features = 240

# Number of subclasses in each class (K). Samples of each class are clustered into subclasses for a more accurate classification.
# Instead of training Class-HVs that represent a whole class, Subclass-HVs are trained. 
param.num_subclasses = 8

# Number of iterations in K-means clustering (T). K-means clustering is used to divide classes into subclasses.
param.num_kmeans_iters = 10

# Number of retrain epochs (tao). In each epoch Subclass-HVs are updated according to the wrongly classified samples in hope to boost accuracy.
param.num_retrain_epochs = 3


# Read dataset from file.
train_features, train_class_labels, test_features, test_class_labels = read_mat.load_data()

# Use less features to train.
num_total_features = 720
train_features = [sample_features[0:num_total_features:num_total_features//param.num_features] for sample_features in train_features]
test_features = [sample_features[0:num_total_features:num_total_features//param.num_features] for sample_features in test_features]


# TRAINING:

# Generate Level-HV Dictionary and Feature-HV Dictionary.
start_time = time.time()
print("Generating Level-HV Dictionary and Feature-HV Dictionary...")
level_hv_dict_, feature_hv_dict_ = hdc.level_hv_dict(), hdc.feature_hv_dict()
print("\t- Level-HV Dictionary and Feature-HV Dictionary Generated.\n")

# Encode training Sample-HV.
print("Encoding train Sample-HVs from train data...")
train_sample_hvs = hdc.encode_sample_hvs(train_features, level_hv_dict_, feature_hv_dict_)
print("\t- Train Sample-HVs encoded and stored.")
encoder_time = time.time() - start_time
print(f"\t- Encoder time in training: {encoder_time: .2f}s.\n")

# Generate subclasses.
start_time = time.time()
print("Generating subclasses...")
subclasses, train_subclass_labels = hdc.generate_subclass(train_sample_hvs, train_class_labels)
print("\t- subclasses generated.")
item_memory_time = time.time() - start_time
print(f"\t- Item Memory time in training: {item_memory_time: .2f}s.\n")
train_time = encoder_time + item_memory_time
print(f'Initial training complete. Total train time: {train_time: .2f}s.\n')


# TESTING:

# Encode testing Sample-HV.
start_time = time.time()
print("Encoding test Sample-HVs from test data...")
test_sample_hvs = hdc.encode_sample_hvs(test_features, level_hv_dict_, feature_hv_dict_)
print("\t- Test Sample-HVs encoded and stored.")
encoder_time = time.time() - start_time
print(f"\t- Encoder time in testing: {encoder_time: .2f}.\n")

# Obtain testing accuracy
start_time = time.time()
print("Calculating testing classification accuracy...")
test_accuracy = hdc.accuracy(test_sample_hvs, test_class_labels, subclasses)
print(f"\t- Classification accuracy: {test_accuracy * 100: .2f}% (test).")
item_memory_time = time.time() - start_time
print(f"\t- Item Memory time in testing: {item_memory_time: .2f}s.\n")
print(f'Testing complete. Test time per 100 samples: {(encoder_time + item_memory_time) / len(test_features) * 100: .2f}s.\n')


# RETRAINING:

# Retrain Subclass-HVs according to the wrongly classified samples.
for epoch in range(param.num_retrain_epochs):
    start_time = time.time()
    print(f"Retraining epoch {epoch: d}: Retraining Subclass-HVs...")
    real_train_accuracy, arithmetic_train_accuracy = hdc.retrain_subclass_hv(train_sample_hvs, train_subclass_labels, subclasses)
    retrain_time = time.time() - start_time
    train_time += retrain_time
    test_accuracy = hdc.accuracy(test_sample_hvs, test_class_labels, subclasses)
    print(f"\t- Classification accuracy: {real_train_accuracy * 100: .2f}% (train), {test_accuracy * 100: .2f}% (test).")
    print(f"\t- Retrain time: {retrain_time: .2f}s.\n")

print(f'Retrain complete. Total train time: {train_time: .2f}s.')
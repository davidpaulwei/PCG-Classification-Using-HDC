import numpy as np

# This python file implements the Hyperdimensional Computing (HDC) functions illustrated in the paper.

class param:
    
    """hyperparams for multi-feature HDC classification project.
       the values of these hyperparams are set in "main.py"."""
    
    # Hypervectors' dimensionality (D).
    dim: int

    # Features' value range. In "read_mat.py" the values are normalized into the range of [0, 1].
    value_range: tuple

    # Number of classes (J). The goal of this task is to classify the samples into these classes.
    num_classes: int
    
    # Number of levels (M), or number of Level-HVs. Level-HVs are used to represent the values of features.
    num_levels: int

    # Number of features (N). Features are represented by Feature-HVs.
    num_features: int

    # Number of subclasses in each class (K). Samples of each class are clustered into subclasses for a more accurate classification.
    # Instead of training Class-HVs that represent a whole class, Subclass-HVs are trained. 
    num_subclasses: int

    # Number of iterations in K-means clustering (T). K-means clustering is used to divide classes into subclasses.
    num_kmeans_iters: int

    # Number of retrain epochs (tao). In each epoch Subclass-HVs are updated according to the wrongly classified samples in hope to boost accuracy.
    num_retrain_epochs: int

class HV_Set:
    pass

class HV_Set(object):

    """Hypervector Set Class that stores the sum and count of a set of hypervectors.
       - When obtaining the bundling (majority function) result of a large set of hypervectors,
         instead of storing every vector in the set, which may lead to a large memory usage,
         we can only record the sum and count of these vectors to obtain the same result."""

    def __init__(self) -> None:
        """Create an empty hypervector set."""
        self.sum = np.zeros(param.dim).astype(int) # The 'sum' attribute stores the arithmetic sum of hypervectors in this set.
        self.count = 0 # The 'count' attribute stores the number of hypervectors in this set.
    
    def add(self, hv: np.ndarray) -> None:
        """Add hypervector to set."""
        self.sum += hv
        self.count += 1
    
    def sub(self, hv: np.ndarray) -> None:
        """Subtract hypervector from set."""
        self.sum -= hv
        self.count -= 1

    def add_set(self, set: HV_Set) -> None:
        """Add all hypervectors from another hv-set to this set."""
        if set.count == 0:  return
        self.sum += set.sum
        self.count += set.count
    
    def bundle(self) -> np.ndarray:
        """Bundle all hypervectors in this set using majority function."""
        if self.count == 0: return hv()
        avg = self.sum / self.count
        bundle_ = np.where(avg > 0, 1, -1)
        # When an even number of hypervectors are bundled, the average float vector may contains elements of '0'. 
        # These elements are snapped into '1' or '-1' by random noise.
        if self.count % 2 == 0: 
            noise = np.where(avg == 0, np.random.choice([0, 2], size = len(avg)), 0)
            bundle_ += noise
        return bundle_

def hv() -> np.ndarray:
    """Generates a random hypervector."""
    return np.random.choice([-1, 1], param.dim)

def feature_hv_dict() -> list:
    """Generates the Feature-HV dictionary. Each randomly-generated Feature-HV is unique and represents a feature."""
    return [hv() for _ in range(param.num_features)]

def level_hv_dict() -> list:
    """Generates the Level-HV dictionary. The features' continuous values are divided into levels, with each one corresponding to a Level-HV.
       - Neighbouring levels have similar Level-HVs, while levels remote to each other have dissimilar Level-HVs.
       - e.g., L[0] has a high positive correlation with L[1], almost no correlation with L[num_levels / 2], and high negative correlation with L[num_levels - 1]."""
    level_hv_dict_ = [hv()] # Level-HV dictionary stores the Level-HVs for each level, with the first Level-HV generated randomly.
    rand_order = np.random.permutation(param.dim)[:param.dim - param.dim % (param.num_levels - 1)].reshape(param.num_levels - 1, -1) # rand_order will decide which elements will be flipped in the previous Level-HV to generate the next Level-HV.
    for level in range(1, param.num_levels):
        # obtain the next Level-HV by flipping parts of the previous Level-HV:
        next_level_hv = level_hv_dict_[-1].copy()
        next_level_hv[rand_order[level - 1]] *= -1
        level_hv_dict_.append(next_level_hv)
    return level_hv_dict_

def get_level(value: int) -> int:
    """obtain level for given value."""
    level_length = (param.value_range[1] - param.value_range[0]) / param.num_levels
    level_idx = int((value - param.value_range[0]) // level_length)
    if level_idx == param.num_levels:   level_idx -= 1
    return level_idx

def encode_sample_hv(features: list, level_hv_dict_: list, feature_hv_dict_: list) -> np.ndarray:
    """generate Sample-HV that represent the sample's overall feature.
       - "features" should be a list containing the sample's features.
       - "level_hv_dict_" and "feature_hv_dict_" should be generated using functions "level_hv_dict()" and "feature_hv_dict()", respectively."""
    feature_value_pairs = HV_Set() # Feature-HVs and Level-HVs are binded into feature-value-pairs. Feature-value-pairs are bundled to generate Sample-HV.
    for feature_idx, feature_value in enumerate(features):
        feature_hv = feature_hv_dict_[feature_idx] # Feature-HV represents the which feature this is.
        level_hv = level_hv_dict_[get_level(feature_value)] # Level-HV represents the feature's value.
        feature_value_pairs.add(feature_hv * level_hv) # bind the Feature-HV and Level-HV using multiplication to generate a feature-value-pair.
    return feature_value_pairs.bundle() # bundle the feature-value-pairs to generate the single Sample-HDV that can represent the overall features of the sample.

def encode_sample_hvs(samples_features: list, level_hv_dict_: list, feature_hv_dict_: list) -> list:
    return [encode_sample_hv(sample_features, level_hv_dict_, feature_hv_dict_) for sample_features in samples_features]

def classify(sample_hv: np.ndarray, subclass_hvs_: list) -> int:
    """classify sample by finding the Subclass-HV with maximum similarity. Simularity is measured by hamming distance."""
    hamming_distance = lambda hv1, hv2: np.sum(hv1 == hv2) / param.dim
    distances = [hamming_distance(sample_hv, subclass_hv) for subclass_hv in subclass_hvs_]
    return np.array(distances).argmax()

def generate_subclass(sample_hvs: list, class_labels: list) -> tuple[list, list]:
    """Divide class into subclass using K-means clustering; Translate class labels into subclass labels.
       - The same index of "sample_hv" list and "class_labels" list belongs to the same PCG sample.
       - Subclasses from the same class has neighbouring labels: class_label = subclass_label // num_subclasses.
       - Return the subclass list and the samples' subclass labels."""
    # Format of "sample_hvs_by_class": [[[Sample_HVs in Class 0], [Class labels in Class 0]], [[Sample_HVs in Class 1], [Class labels in Class 1]],...].
    sample_hvs_by_class = [[[], []] for _ in range(param.num_classes)]
    for sample_idx, (sample_hv, class_label) in enumerate(zip(sample_hvs, class_labels)):
        sample_hvs_by_class[class_label][0].append(sample_hv)
        sample_hvs_by_class[class_label][1].append(sample_idx)
    subclasses, subclass_labels = [], [None for _ in range(len(sample_hvs))]
    for class_idx, [sample_hvs_in_this_class, sample_idxs] in enumerate(sample_hvs_by_class):
        subclasses_in_this_class, subclass_labels_in_this_class = k_means_clustering(sample_hvs_in_this_class)
        subclasses += subclasses_in_this_class
        for sample_idx, subclass_label in zip(sample_idxs, subclass_labels_in_this_class):
            subclass_labels[sample_idx] = class_idx * param.num_subclasses + subclass_label
    return subclasses, subclass_labels

def retrain_subclass_hv(sample_hvs: list, subclass_labels: list, subclasses: list) -> tuple[float, float]:
    """obtain the accuracy of predicting sample's class (real accuracy) and subclass (arithmetic accuracy) while retraining Subclass-HVs."""
    num_arithmetic_errors, num_real_errors, error_samples_by_subclass = 0, 0, [HV_Set() for _ in range(param.num_classes * param.num_subclasses)] # the wrongly classified samples are stored in "error_sample_sets".
    subclass_hvs = [subclass.bundle() for subclass in subclasses] # obtain Subclass-HVs by bundling all the Sample-HVs in that subclass.
    for sample_hv, subclass_label in zip(sample_hvs, subclass_labels):
        predict = classify(sample_hv, subclass_hvs)
        if predict != subclass_label: # this sample is wrongly classified.
            num_arithmetic_errors += 1
            error_samples_by_subclass[subclass_label].add(sample_hv)
            error_samples_by_subclass[predict].sub(sample_hv)
            if predict // param.num_subclasses != subclass_label // param.num_subclasses:
                num_real_errors += 1
    for subclass, error_samples_in_this_subclass in zip(subclasses, error_samples_by_subclass):
        subclass.add_set(error_samples_in_this_subclass) 
    real_accuracy = 1 - num_real_errors / len(sample_hvs)
    arithmetic_accuracy = 1 - num_arithmetic_errors / len(sample_hvs)
    return real_accuracy, arithmetic_accuracy

def accuracy(sample_hvs: list, class_labels: list, subclasses: list) -> float:
    """obtain the testing accuracy of the 'classify' function using Class-HDVs."""
    num_errors = 0
    subclass_hvs = [subclass.bundle() for subclass in subclasses] # obtain Subclass-HVs by bundling all the Sample-HVs in that subclass.
    for sample_hv, class_label in zip(sample_hvs, class_labels):
        if classify(sample_hv, subclass_hvs) // param.num_subclasses != class_label:
            num_errors += 1
    accuracy_ = 1 - num_errors / len(sample_hvs)
    return accuracy_


def k_means_clustering(sample_hvs: list) -> tuple[list, list]:
    """Cluster Sample-HVs into subclasses;
       Returns:
       - "subclasses": A list of HV_Set objects. Each HV_Set object represent all the Sample-HVs that belongs to the subclass. 
            The ".bundle()" result of each HV_Set object is the Subclass-HV of that subclass.
       - "subclass_labels": List of subclass labels. Each label range from [0, param.num_subclasses).
            The same index of the "sample_hvs" list and the "subclass_labels" list belongs to the same PCG sample."""
    subclasses = [HV_Set() for _ in range(param.num_subclasses)]
    for sample_idx, sample_hv in enumerate(sample_hvs):
        subclasses[sample_idx % param.num_subclasses].add(sample_hv) # randomly assign Sample-HVs to subclasses
    subclass_hvs_ = [subclass.bundle() for subclass in subclasses] # bundle all Sample-HVs in each subclass to generate Subclass-HVs.
    for _ in range(param.num_kmeans_iters):
        subclasses = [HV_Set() for _ in range(param.num_subclasses)] # reinitialize subclass.
        for sample_hv in sample_hvs:
            subclasses[classify(sample_hv, subclass_hvs_)].add(sample_hv) # samples are classified into the nearest Subclass.
        subclass_hvs_ = [subclass.bundle() for subclass in subclasses]
    subclass_labels = [classify(sample_hv, subclass_hvs_) for sample_hv in sample_hvs]
    return subclasses, subclass_labels
     
        




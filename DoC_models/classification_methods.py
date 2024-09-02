import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score


# FUNCTION1 - Train the SVM classifier
def train_classif(x_train, y_train, c_val=0.01, kernel_type='linear', gamma_val=7.5, deg=3):
    # SVM classifier
    classifier = SVC(kernel=kernel_type, random_state=1, probability=True, C=c_val, gamma=gamma_val, class_weight='balanced', degree=deg)
    classifier.fit(x_train, y_train)
    return classifier


# FUNCTION2 - Use the trained SVM classifier to classify the test data
def predict_class(x_to_classify, classifier):
    predicted_classes = classifier.predict(x_to_classify)
    return predicted_classes


# FUNCTION3 - Evaluate classifier accuracy
def eval_accuracy(x_test, y_real, classifier):
    accuracy = classifier.score(x_test, y_real)
    #accuracy = accuracy_score(y_real, x_test)
    return accuracy


# FUNCTION4 - Evaluate classifier precision and recall
def eval_classif(y_pred, y_real):
    precision = precision_score(y_real, y_pred, average='weighted', zero_division=1.0)
    recall = recall_score(y_real, y_pred, average='weighted', zero_division=1.0)
    f1 = f1_score(y_real, y_pred, average='weighted', zero_division=1.0)
    return precision, recall, f1


# FUNCTION5 - Build training and test sets
def build_training_test_sets_classif(dataset, dataset_y, prop=0.75, criteria='FC', TR=2):
    num_patients = dataset.shape[0]
    num_rois = dataset.shape[1]
    time = dataset.shape[2]
    # Shuffle and select prop % in training and the rest in test
    x_train = np.zeros([int(np.floor(num_patients*prop)), num_rois*num_rois])
    x_test = np.zeros([int(np.ceil(num_patients * (1-prop))), num_rois*num_rois])
    training_patients = np.random.choice(num_patients, int(np.floor(num_patients*prop)), replace=False)
    y_train = np.zeros(int(np.floor(num_patients*prop)))
    test_patients = np.setdiff1d(range(0, num_patients), training_patients)
    y_test = np.zeros(int(np.ceil(num_patients * (1-prop))))
    # -------------- Functional Connectivity GLOBAL ---------------------- #
    if criteria == 'FC':
        for pat_idx in range(0, len(training_patients)):
            pat_num = training_patients[pat_idx]
            y_train[pat_idx] = dataset_y[pat_num]
            corr_mat = np.corrcoef(dataset[pat_num, :, :], rowvar=True)
            x_train[pat_idx, :] = corr_mat.flatten()
        for pat_idx in range(0, len(test_patients)):
            pat_num = test_patients[pat_idx]
            y_test[pat_idx] = dataset_y[pat_num]
            corr_mat = np.corrcoef(dataset[pat_num, :, :], rowvar=True)
            x_test[pat_idx, :] = corr_mat.flatten()
    # -------------- Effective Connectivity ---------------------- #
    if criteria == 'EC':
        for pat_idx in range(0, len(training_patients)):
            pat_num = training_patients[pat_idx]
            y_train[pat_idx] = dataset_y[pat_num]
            x_train[pat_idx, :] = dataset[pat_idx, :, :].flatten()
        for pat_idx in range(0, len(test_patients)):
            pat_num = test_patients[pat_idx]
            y_test[pat_idx] = dataset_y[pat_num]
            x_test[pat_idx, :] = dataset[pat_idx, :, :].flatten()
    # -------------- Functional Connectivity DYNAMIC ---------------------- #
    if criteria == 'FCD':
        # Cut time in slices with overlap
        chunk_siz = 60/TR
        chunk_overlap = 40/TR
        num_chunks = int(np.floor(time/(chunk_siz-chunk_overlap)))
        x_train = np.zeros([int(np.floor(num_patients * prop)), num_chunks * num_chunks])
        x_test = np.zeros([int(np.ceil(num_patients * (1 - prop))), num_chunks * num_chunks])
        for pat_idx in range(0, len(training_patients)):
            pat_num = training_patients[pat_idx]
            y_train[pat_idx] = dataset_y[pat_num]
            FCD_mat = np.zeros([num_chunks, num_chunks])
            corr_mat = np.zeros([num_chunks, num_rois, num_rois])
            corr_mat_flat = np.zeros([num_chunks, num_rois*num_rois])
            for t in range(0, num_chunks):
                time_window_in = int(t*(chunk_siz-chunk_overlap))
                time_window_out = int(t*(chunk_siz-chunk_overlap)+chunk_siz)
                corr_mat[t, :, :] = np.corrcoef(dataset[pat_num, :, time_window_in:time_window_out], rowvar=True)
                corr_mat_flat[t, :] = corr_mat[t, :, :].flatten()
                x_train[pat_idx,t*num_chunks:(t+1)*num_chunks] = corr_mat_flat[t,:]

            """
            for t in range(0, num_chunks):
                for t2 in range(t+1, num_chunks):
                    FCD_mat[t, t2] = np.corrcoef(corr_mat_flat[t, :], corr_mat_flat[t2, :])[1, 0]
                x_train[pat_idx, t*num_chunks:(t+1)*num_chunks] = FCD_mat[t, :]
            """
                
        for pat_idx in range(0, len(test_patients)):
            pat_num = test_patients[pat_idx]
            y_test[pat_idx] = dataset_y[pat_num]
            FCD_mat = np.zeros([num_chunks, num_chunks])
            corr_mat = np.zeros([num_chunks, num_rois, num_rois])
            corr_mat_flat = np.zeros([num_chunks, num_rois*num_rois])
            for t in range(0, num_chunks):
                time_window_in = t * (chunk_siz - chunk_overlap)
                time_window_out = t * (chunk_siz - chunk_overlap) + chunk_siz
                corr_mat[t, :, :] = np.corrcoef(dataset[pat_num, :, time_window_in:time_window_out], rowvar=True)
                corr_mat_flat[t, :] = corr_mat[t, :, :].flatten() 
                x_train[pat_idx,t*num_chunks:(t+1)*num_chunks] = corr_mat_flat[t,:]
            """
            for t in range(0, num_chunks):
                for t2 in range(t + 1, num_chunks):
                    FCD_mat[t, t2] = np.corrcoef(corr_mat_flat[t, :], corr_mat_flat[t2, :])[1, 0]
                x_test[pat_idx, t * num_chunks:(t + 1) * num_chunks] = FCD_mat[t, :]
            """
               
    if criteria == 'tenet':
        x_train = np.zeros([int(np.floor(num_patients * prop)), time * time])
        x_test = np.zeros([int(np.ceil(num_patients * (1 - prop))), time * time])
        # Just take the direct time correlation matrices and flatten to a vector
        # Shuffle and select prop % in training and the rest in test
        training_patients = np.random.choice(num_patients, int(np.floor(num_patients * prop)), replace=False)
        y_train = np.zeros(int(np.floor(num_patients * prop)))
        test_patients = np.setdiff1d(range(0, num_patients), training_patients)
        y_test = np.zeros(int(np.ceil(num_patients * (1 - prop))))
        for pat_idx in range(0, len(training_patients)):
            pat_num = training_patients[pat_idx]
            y_train[pat_idx] = dataset_y[pat_num]
            corr_mat = np.corrcoef(dataset[pat_num, :, :], rowvar=False)
            corr_mat_inv = np.corrcoef(dataset[pat_num, :, ::-1], rowvar=False)
            info_direct = -0.5 * np.log(1 - corr_mat * corr_mat)
            info_reversed = -0.5 * np.log(1 - corr_mat_inv * corr_mat_inv)
            mutual_info_matrix = abs(info_direct - info_reversed)
            for t_idx in range(0, time):
                x_train[pat_idx, t_idx * time:(t_idx + 1) * time] = mutual_info_matrix[t_idx, :]
        for pat_idx in range(0, len(test_patients)):
            pat_num = test_patients[pat_idx]
            y_test[pat_idx] = dataset_y[pat_num]
            corr_mat = np.corrcoef(dataset[pat_num, :, :], rowvar=False)
            corr_mat_inv = np.corrcoef(dataset[pat_num, :, ::-1], rowvar=False)
            info_direct = -0.5 * np.log(1 - corr_mat * corr_mat)
            info_reversed = -0.5 * np.log(1 - corr_mat_inv * corr_mat_inv)
            mutual_info_matrix = abs(info_direct - info_reversed)
            for t_idx in range(0, time):
                x_test[pat_idx, t_idx * time:(t_idx + 1) * time] = mutual_info_matrix[t_idx, :]
    # Remove the NaN of Inf values
    x_train[np.isnan(x_train)] = 0
    x_test[np.isnan(x_test)] = 0
    x_train[np.isposinf(x_train)] = 1
    x_test[np.isposinf(x_test)] = 1
    x_train[np.isneginf(x_train)] = -1
    x_test[np.isneginf(x_test)] = -1
    return x_train, x_test, y_train, y_test




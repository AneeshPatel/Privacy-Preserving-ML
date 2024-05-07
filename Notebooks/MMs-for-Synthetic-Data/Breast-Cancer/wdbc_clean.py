#!/usr/bin/env python3
#
# This code has been produced by a free evaluation version of Brainome(tm).
# Portions of this code copyright (c) 2019-2022 by Brainome, Inc. All Rights Reserved.
# Brainome, Inc grants an exclusive (subject to our continuing rights to use and modify models),
# worldwide, non-sublicensable, and non-transferable limited license to use and modify this
# predictor produced through the input of your data:
# (i) for users accessing the service through a free evaluation account, solely for your
# own non-commercial purposes, including for the purpose of evaluating this service, and
# (ii) for users accessing the service through a paid, commercial use account, for your
# own internal  and commercial purposes.
# Please contact support@brainome.ai with any questions.
# Use of predictions results at your own risk.
#
# Output of Brainome v2.0-172-prod.
# Invocation: brainome wdbc_clean.csv -o wdbc_clean.py -e 5 -y
# Total compiler execution time: 0:02:16.85. Finished on: May-01-2024 14:11:03.
# This source code requires Python 3.
#
"""

[01;1mPredictor:[0m                        
    Classifier Type:              Neural Network
    System Type:                  Binary classifier
    Training / Validation Split:  50% : 50%
    Accuracy:
      Best-guess accuracy:        62.74%
      Training accuracy:          98.94% (281/284 correct)
      Validation Accuracy:        94.03% (268/285 correct)
      Combined Model Accuracy:    96.48% (549/569 correct)


    Model Capacity (MEC):        45    bits
    Generalization Ratio:         5.95 bits/bit
    Percent of Data Memorized:    34.26%
    Resilience to Noise:          -0.80 dB




    Training Confusion Matrix:
              Actual | Predicted
              ------ | ---------
                 0.0 |  106    0 
                 1.0 |    3  175 

    Validation Confusion Matrix:
              Actual | Predicted
              ------ | ---------
                 0.0 |   98    8 
                 1.0 |    9  170 

    Training Accuracy by Class:
              target |   TP   FP   TN   FN     TPR      TNR      PPV      NPV       F1       TS 
              ------ | ---- ---- ---- ---- -------- -------- -------- -------- -------- --------
                 0.0 |  106    3  175    0  100.00%   98.31%   97.25%  100.00%   98.60%   97.25%
                 1.0 |  175    0  106    3   98.31%  100.00%  100.00%   97.25%   99.15%   98.31%

    Validation Accuracy by Class:
              target |   TP   FP   TN   FN     TPR      TNR      PPV      NPV       F1       TS 
              ------ | ---- ---- ---- ---- -------- -------- -------- -------- -------- --------
                 0.0 |   98    9  170    8   92.45%   94.97%   91.59%   95.51%   92.02%   85.22%
                 1.0 |  170    8   98    9   94.97%   92.45%   95.51%   91.59%   95.24%   90.91%



"""

import sys
import math
import argparse
import csv
import binascii
import faulthandler
import json
try:
    import numpy as np  # For numpy see: http://numpy.org
except ImportError as e:
    print("This predictor requires the Numpy library. Please run 'python3 -m pip install numpy'.", file=sys.stderr)
    raise e
try:
    from scipy.sparse import coo_matrix
    report_cmat = True
except ImportError:
    print("Note: If you install scipy (https://www.scipy.org) this predictor generates a confusion matrix. Try 'python3 -m pip install scipy'.", file=sys.stderr)
    report_cmat = False

IOBUFF = 100000000
sys.setrecursionlimit(1000000)
random_filler_value = 'ba8db6eb493e918dd0b9b7facc14a63caf0749d4510adbd022df4c13b8ba8f5f'
TRAINFILE = ['wdbc_clean.csv']
mapping = {'0.0': 0, '1.0': 1}
ignorelabels = []
ignorecolumns = []
target = 'target'
target_column = 30
important_idxs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
expected_feature_cols = 30
classifier_type = 'NN'
num_attr = 30
n_classes = 2
model_cap = 45
w_h = np.array([[0.48465320467948914, 0.8597012758255005, -0.34232097864151, 2.473917007446289, -4.432367324829102, 0.0518609881401062, 9.227192878723145, -3.264979124069214, 13.6824951171875], [-0.23024386167526245, -0.057245537638664246, -3.240546226501465, -6.184139251708984, -3.602628231048584, 8.945374488830566, 13.279645919799805, -0.3059806227684021, -4.362585067749023], [0.23911356925964355, 0.6006734371185303, -1.3571034669876099, -5.851751804351807, 8.543265342712402, 3.833644151687622, -1.3358643054962158, -1.9739139080047607, -7.410891056060791], [0.6715981364250183, 2.488772392272949, -0.9674664735794067, 5.699151039123535, -4.180453300476074, -11.057384490966797, -5.371803283691406, -8.438423156738281, 15.791698455810547]])
b_h = np.array([-6.434633731842041, -6.046297550201416, -3.6580286026000977, -2.7712242603302])
w_o = np.array([[-0.809596836566925, 0.4079442024230957, 0.46169453859329224, -0.6877016425132751]])
b_o = np.array(-0.8942448496818542)


class PredictorError(Exception):

    def __init__(self, msg, code):
        self.msg = msg
        self.code = code

    def __str__(self):
        return self.msg
def __transform(X):
    mean = np.array([14.179017605633803, 19.344049295774653, 92.20179577464782, 659.3651408450703, 0.09615630281690137, 0.10225154929577468, 0.0882103088028169, 0.04875724647887325, 0.1818457746478872, 0.06258595070422539, 0.39830492957746483, 1.2493492957746468, 2.799492957746478, 39.27271126760564, 0.007020165492957747, 0.025292211267605634, 0.03279048098591551, 0.011783538732394365, 0.020754190140845088, 0.003824077816901407, 16.308056338028184, 25.78369718309858, 107.30341549295775, 884.4316901408457, 0.13137014084507032, 0.24827492957746494, 0.2677045669014081, 0.11396331338028173, 0.2907288732394365, 0.08369359154929575])
    components = np.array([[0.0052754532127307205, 0.0027186600141417827, 0.036289120957442036, 0.5179534757100491, 3.66410279950542e-06, 4.294902401968925e-05, 8.407321548735884e-05, 4.947925375820487e-05, 9.396036688276433e-06, -2.7515730195499646e-06, 0.0002811983491057373, -8.736575394313964e-05, 0.0019779348729372665, 0.04736596935004984, -9.474041158685845e-07, 6.0749997164503205e-06, 8.726206202304845e-06, 3.3703227127251597e-06, -1.6263020067580738e-06, -1.0786402794002254e-07, 0.007402629644050535, 0.0038332182106576656, 0.05106190739708281, 0.8517321576460324, 6.7357100541119684e-06, 0.00010821610512996263, 0.00017908245393918117, 7.754370912791702e-05, 2.2599984847769744e-05, 2.3148925205998146e-06], [-0.009725848655860502, 0.003331813132745484, -0.06614077170501054, -0.8513770108038414, 1.6413644843462597e-05, -4.114275044432868e-06, -7.709033626862828e-05, -5.1390450609508516e-05, 1.2541253734984533e-05, 1.794928785849486e-05, 0.00020003953663810488, -0.000437889354642325, 0.0002510844254446497, 0.02532582394240105, -1.4842303438136272e-06, -1.93086450006467e-05, -3.489333813015308e-05, -1.484220582570917e-05, -9.706223796993625e-06, 8.17370188016073e-08, 0.00014692115980474257, 0.014757684623467318, -0.004816552589861425, 0.5194170194876789, 8.600648459741107e-05, 0.00026684488159485394, 0.00017371256906099043, 1.4389020095774563e-05, 0.00015548752158005434, 6.28041519221817e-05], [-0.009118714902386024, 0.007235922410565853, -0.02651641219065817, 8.682885683929586e-05, 0.00020876381618672776, 0.0007655847691545369, 0.001232547259312845, 0.00043283327822902265, 0.0004849161431020136, 0.0001284922743921423, 0.008074411625286165, 0.015448854140126513, 0.057040684363610705, 0.9946384001226013, 9.144577774620558e-05, 0.0004253679244741682, 0.0006784287046416042, 0.0001289444064010489, 0.0002530701460625077, 6.53931673780492e-05, -0.013108253694050952, -0.03752326392769589, -0.04562273108187673, -0.05131867510671531, 3.3336464981948765e-05, -9.165136953559836e-05, 0.00021051932549563286, -0.00016323809543009485, -0.00021974250779152336, 1.0747156900993888e-05], [0.03644719249026508, 0.28084998450857185, 0.35399923219262514, -0.045550359442040805, 0.00037049614873421456, 0.0033837256539465286, 0.0034428606297008486, 0.0013115477409822815, 0.0010645228151893729, 0.00019526055611722117, 0.002875147514586652, 0.01377342182838747, 0.04944181124570206, 0.05544782131836093, -3.9599240483553574e-05, 0.0010361431125054212, 0.0009564923403540277, 0.00024039786035094174, 0.00017386561029363894, 6.478018196603824e-05, 0.06759875045347598, 0.4606514219398296, 0.7536507294902167, -0.03955043723709642, 0.0008062394237384573, 0.012493927513614288, 0.012404693468328681, 0.0033039976009642358, 0.0037026093941996646, 0.0010111777111167983], [0.030060750086751492, -0.5054110001728311, 0.24462843010621044, -0.04152352871455112, 0.0002795259945437927, 0.001389196491626371, 0.0007425607316274675, 0.0006480300279333906, 0.0004522652025742214, 6.952758606511648e-05, 7.530158069103402e-05, -0.04691196852503386, 0.016606243489388956, 0.006726961761881464, -0.00011160597714435165, 9.12062055414481e-05, -0.00011558513187345837, 5.028911398635269e-05, -2.5914895365031767e-05, -1.6184670030913486e-05, 0.042021231660436385, -0.6722061480416442, 0.4751974718153621, -0.009991648023800537, 0.00016907469197856274, 0.0037109352023358896, 0.0022863129618360153, 0.0015500002696667411, 0.0017296605320858095, 0.0001874462062199617], [-0.14295861336885332, -0.06588142087758787, -0.8699668373867969, 0.054620891988355805, -0.000432257278134497, -9.976721119808622e-05, 0.0014597722976637771, -0.00011002264821214208, 0.0003432611243681128, 0.00047346775784316803, 0.004847551092588617, 0.01083942909153789, 0.13744656123679758, -0.013948901318622515, 0.0001799519177509351, 0.0003847637072090864, 0.0006205432092458107, -0.0001482003480473118, 0.00018256973010967096, 9.860568693923134e-05, -0.10439739720945712, 0.022958241887791227, 0.4292645962401459, -0.01953120675704729, 0.0011484274001431318, 0.006015732857070758, 0.006268188649184469, 0.0009765830223237364, 0.0034431551399790807, 0.0011107164360388924], [0.0052090962987768955, 0.8068425758781407, -0.03288477243138107, -0.004228120500241406, -0.0017773483411224005, -0.002734191849334097, -0.0036010905480811733, -0.001850417241912078, -0.0029425865039773434, -0.00047189777359261785, 0.009105047499388818, -0.10460071695325347, 0.10065055909221034, -0.029522827230673766, 4.254423144002899e-05, -0.00032059646354592526, -6.507508223893896e-05, 0.0003579792752681771, 0.0002563543545028247, -3.176175541518704e-05, -0.002214048371496557, -0.5670801981409576, 0.05928133133962522, 0.0017833931287906113, -0.004599090456731584, -0.015810188655181294, -0.01676389079786128, -0.005009777718951756, -0.011428074994403972, -0.002435427317320679], [-0.11189900939468092, -0.02493118251524155, 0.14306573831264305, -0.007286295143727214, 0.0016547847847280018, 0.019467761229564156, 0.01775972392597742, 0.005708136852055116, 0.007924909182552873, 0.004058624915467332, 0.036197638092540424, 0.5719029090832151, 0.7433655227433484, -0.05620730329409137, 0.0009387797677949798, 0.0092292530013566, 0.014173376309889437, 0.004518646090222538, 0.006237266866534635, 0.0016548585172984087, -0.27291152530470314, -0.022567868534804185, -0.07046270187659469, 0.0072408209953020095, -0.003234787519608124, 0.019587301606541783, 0.024528605066988094, 0.0038920930011148588, -0.003325317898011066, 0.0037368395923125463], [-0.3340787635749682, 0.018567230752942872, 0.1312811068614094, -0.0038644664988441713, 0.010309447394278875, 0.0571507728144446, 0.07145800220835159, 0.01989324296319429, 0.015729418089615643, 0.007919092942732579, -0.08416464747547207, -0.1892804615488407, -0.24154837462252338, 0.008407997607081483, 0.0013707040107894279, 0.018265475529849393, 0.02829702827671991, 0.0025960004226314786, 0.0022979621717368836, 0.001851971293295711, -0.8303302962961298, 0.00033709693171432207, 0.033826151876951316, 0.003977164149366989, 0.015632690857401994, 0.1570937302166248, 0.2024084436394802, 0.03386965854385611, 0.033372120537956136, 0.019292544513657563]])
    explained_variance = np.array([420051.99975559313, 7495.543976829927, 299.90683308588115, 51.08549533222469, 34.686227047300115, 2.8303946141766465, 1.675401647338821, 0.24678407427587895, 0.1243735255612289])
    X = X - mean
    X_transformed = np.dot(X, components.T)
    return X_transformed


def __convert(cell):
    value = str(cell)
    if value == random_filler_value:
        value = ''
    try:
        result = int(value)
        return result
    except ValueError:
        try:
            result = float(value)
            if math.isnan(result):
                raise PredictorError('NaN value found. Aborting.', code=1)
            return result
        except ValueError:
            result = (binascii.crc32(value.encode('utf8')) % (1 << 32))
            return result
        except Exception as e:
            raise e


def __get_key(val, dictionary):
    if dictionary == {}:
        return val
    for key, value in dictionary.items():
        if val == value:
            return key
    if val not in dictionary.values():
        raise PredictorError(f"Label {val} key does not exist", code=2)


def __confusion_matrix(y_true, y_pred, json):
    stats = {}
    labels = np.array(list(mapping.keys()))
    sample_weight = np.ones(y_true.shape[0], dtype=np.int64)
    for class_i in range(n_classes):
        class_i_label = __get_key(class_i, mapping)
        stats[int(class_i)] = {}
        class_i_indices = np.argwhere(y_true == class_i_label)
        not_class_i_indices = np.argwhere(y_true != class_i_label)
        # None represents N/A in this case
        stats[int(class_i)]['TP'] = TP = int(np.sum(y_pred[class_i_indices] == class_i_label)) if class_i_indices.size > 0 else None
        stats[int(class_i)]['FN'] = FN = int(np.sum(y_pred[class_i_indices] != class_i_label)) if class_i_indices.size > 0 else None
        stats[int(class_i)]['TN'] = TN = int(np.sum(y_pred[not_class_i_indices] != class_i_label)) if not_class_i_indices.size > 0 else None
        stats[int(class_i)]['FP'] = FP = int(np.sum(y_pred[not_class_i_indices] == class_i_label)) if not_class_i_indices.size > 0 else None
        if TP is None or FN is None or (TP + FN == 0):
            stats[int(class_i)]['TPR'] = None
        else:
            stats[int(class_i)]['TPR'] = (TP / (TP + FN))
        if TN is None or FP is None or (TN + FP == 0):
            stats[int(class_i)]['TNR'] = None
        else:
            stats[int(class_i)]['TNR'] = (TN / (TN + FP))
        if TP is None or FP is None or (TP + FP == 0):
            stats[int(class_i)]['PPV'] = None
        else:
            stats[int(class_i)]['PPV'] = (TP / (TP + FP))
        if TN is None or FN is None or (TN + FN == 0):
            stats[int(class_i)]['NPV'] = None
        else:
            stats[int(class_i)]['NPV'] = (TN / (TN + FN))
        if TP is None or FP is None or FN is None or (TP + FP + FN == 0):
            stats[int(class_i)]['F1'] = None
        else:
            stats[int(class_i)]['F1'] = ((2 * TP) / (2 * TP + FP + FN))
        if TP is None or FP is None or FN is None or (TP + FP + FN == 0):
            stats[int(class_i)]['TS'] = None
        else:
            stats[int(class_i)]['TS'] = (TP / (TP + FP + FN))

    if not report_cmat:
        return np.array([]), stats

    label_to_ind = {label: i for i, label in enumerate(labels)}
    y_pred = np.array([label_to_ind.get(x, n_classes + 1) for x in y_pred])
    y_true = np.array([label_to_ind.get(x, n_classes + 1) for x in y_true])

    ind = np.logical_and(y_pred < n_classes, y_true < n_classes)
    y_pred = y_pred[ind]
    y_true = y_true[ind]
    sample_weight = sample_weight[ind]

    cm = coo_matrix((sample_weight, (y_true, y_pred)), shape=(n_classes, n_classes), dtype=np.int64).toarray()
    with np.errstate(all='ignore'):
        cm = np.nan_to_num(cm)

    return cm, stats


def __preprocess_and_clean_in_memory(arr):
    clean_arr = np.zeros((len(arr), len(important_idxs)))
    for i, row in enumerate(arr):
        try:
            row_used_cols_only = [row[i] for i in important_idxs]
        except IndexError:
            error_str = f"The input has shape ({len(arr)}, {len(row)}) but the expected shape is (*, {len(ignorecolumns) + len(important_idxs)})."
            if len(arr) == num_attr and len(arr[0]) != num_attr:
                error_str += "\n\nNote: You may have passed an input directly to 'preprocess_and_clean_in_memory' or 'predict_in_memory' "
                error_str += "rather than as an element of a list. Make sure that even single instances "
                error_str += "are enclosed in a list. Example: predict_in_memory(0) is invalid but "
                error_str += "predict_in_memory([0]) is valid."
            raise PredictorError(error_str, 3)
        clean_arr[i] = [float(__convert(field)) for field in row_used_cols_only]
    return clean_arr


def __classify(arr, return_probabilities=False):
    h = np.dot(arr, w_h.T) + b_h
    relu = np.maximum(h, np.zeros_like(h))
    out = np.dot(relu, w_o.T) + b_o
    if return_probabilities:
        exp_o = np.zeros((out.shape[0],))
        idxs_negative = np.argwhere(out < 0.).reshape(-1)
        if idxs_negative.shape[0] > 0:
            exp_o[idxs_negative] = 1. - np.exp(-np.fmax(out[idxs_negative], 0)).reshape(-1) / (1. + np.exp(-np.abs(out[idxs_negative]))).reshape(-1)
        idxs_positive = np.argwhere(out >= 0.).reshape(-1)
        if idxs_positive.shape[0] > 0:
            exp_o[idxs_positive] = np.exp(np.fmin(out[idxs_positive], 0)).reshape(-1) / (1. + np.exp(-np.abs(out[idxs_positive]))).reshape(-1)
        exp_o = exp_o.reshape(-1, 1)
        output = np.concatenate((1. - exp_o, exp_o), axis=1)
    else:
        output = (out >= 0).astype('int').reshape(-1)
    return output



def __validate_kwargs(kwargs):
    for key in kwargs:

        if key not in ['return_probabilities']:
            raise PredictorError(f'{key} is not a keyword argument for Brainome\'s {classifier_type} predictor. Please see the documentation.', 4)


def __validate_data(row_or_arr, validate, row_num=None):
    if validate:
        expected_columns = expected_feature_cols + 1
    else:
        expected_columns = expected_feature_cols

    input_is_array = isinstance(row_or_arr, np.ndarray)
    n_cols = row_or_arr.shape[1] if input_is_array else len(row_or_arr)

    if n_cols != expected_columns:

        if row_num is None:
            err_str = f"Your data contains {n_cols} columns but {expected_columns} are required."
        else:
            err_str = f"At row {row_num}, your data contains {n_cols} columns but {expected_columns} are required."

        if validate:
            err_str += " The predictor's validate() method works on data that has the same columns in the same order as were present in the training CSV."
            err_str += " This includes the target column and features that are not used by the model but existed in the training CSV."
            if n_cols == 1 + len(important_idxs):
                err_str += f" We suggest confirming that the {expected_feature_cols - len(important_idxs)} unused features are present in the data."
            elif n_cols == len(important_idxs):
                err_str += f" We suggest confirming that the {expected_feature_cols - len(important_idxs)} unused features are present in the data as well as the target column. "
            elif n_cols == len(important_idxs) + len(ignore_idxs):
                err_str += " We suggest confirming that the target column present in the data. "
            err_str += " To make predictions, see the predictor's predict() method."
        else:
            err_str += " The predictor's predict() method works on data that has the same feature columns in the same relative order as were present in the training CSV."
            err_str += " This DOES NOT include the target column but DOES include features that are not used by the model but existed in the training CSV."
            if n_cols == 1 + len(important_idxs):
                err_str += f" We suggest confirming that the {expected_feature_cols - len(important_idxs)} unused features are present in the data and that the target column is not present."
            elif n_cols == len(important_idxs):
                err_str += f" We suggest confirming that the {expected_feature_cols - len(important_idxs)} unused features are present in the data."
            elif n_cols == 1 + len(important_idxs) + len(ignore_idxs):
                err_str += " We suggest confirming that the target column is not present."
            err_str += " To receive a performance summary, instead of make predictions, see the predictor's validate() method."

        raise PredictorError(err_str, 5)

    else:

        if not input_is_array:
            return row_or_arr


def __write_predictions(arr, header, headerless, trim, outfile=None):
    predictions = predict(arr)
    buff = []

    if not headerless:
        if trim:
            header = ','.join([header[i] for i in important_idxs] + ['Prediction'])
        else:
            header = ','.join(header.tolist() + ['Prediction'])
        if outfile is None:
            print(header)
        else:
            print(header, file=outfile)

    for row, prediction in zip(arr, predictions):
        if trim:
            row = [f'"{row[i]}",' if ',' in row[i] else f'{row[i]},' for i in important_idxs]
        else:
            row = [f'"{field}",' if ',' in field else f'{field},' for field in row]
        row.append(prediction)
        buff.extend(row)
        if len(buff) >= IOBUFF:
            if outfile is None:
                print(''.join(buff))
            else:
                print(''.join(buff), file=outfile)
            buff = []
        else:
            buff.append('\n')
    if len(buff) > 0:
        if outfile is None:
            print(''.join(buff))
        else:
            print(''.join(buff), file=outfile)


def load_data(csvfile, headerless, validate):
    """
    Parameters
    ----------
    csvfile : str
        The path to the CSV file containing the data.

    headerless : bool
        True if the CSV does not contain a header.

    validate : bool
        True if the data should be loaded to be used by the predictor's validate() method.
        False if the data should be loaded to be used by the predictor's predict() method.

    Returns
    -------
    arr : np.ndarray
        The data (observations and labels) found in the CSV without any header.

    data : np.ndarray or NoneType
        None if validate is False, otherwise the observations (data without the labels) found in the CSV.

    labels : np.ndarray or NoneType
        None if the validate is False, otherwise the labels found in the CSV.

    header : np.ndarray or NoneType
        None if the CSV is headerless, otherwise the header.
    """

    with open(csvfile, 'r', encoding='utf-8') as csvinput:
        arr = np.array([__validate_data(row, validate, row_num=i) for i, row in enumerate(csv.reader(csvinput)) if row != []], dtype=str)
    if headerless:
        header = None
    else:
        header = arr[0]
        arr = arr[1:]
    if validate:
        labels = np.char.strip(arr[:, target_column], chars=" \"\'")
        feature_columns = [i for i in range(arr.shape[1]) if i != target_column]
        data = arr[:, feature_columns]
    else:
        data, labels = None, None

    if validate and ignorelabels != []:
        idxs_to_keep = np.argwhere(np.logical_not(np.isin(labels, ignorelabels))).reshape(-1)
        labels = labels[idxs_to_keep]
        data = data[idxs_to_keep]

    return arr, data, labels, header


def predict(arr, remap=True, **kwargs):
    """
    Parameters
    ----------
    arr : list[list]
        An array of inputs to be cleaned by 'preprocess_and_clean_in_memory'. This
        should contain all the features that were present in the training data,
        regardless of whether or not they are used by the model, with the same
        relative order as in the training data. There should be no target column.


    remap : bool
        If True and 'return_probs' is False, remaps the output to the original class
        label. If 'return_probs' is True this instead adds a header indicating which
        original class label each column of output corresponds to.

    **kwargs :
        return_probabilities : bool
            If true, return class membership probabilities instead of classifications.

    Returns
    -------
    output : np.ndarray

        A numpy array of
            1. Class predictions if 'return_probabilities' is False.
            2. Class probabilities if 'return_probabilities' is True.

    """
    if not isinstance(arr, np.ndarray) and not isinstance(arr, list):
        raise PredictorError(f'Data must be provided to \'predict\' and \'validate\' as a list or np.ndarray, but an input of type {type(arr).__name__} was found.', 6)
    if isinstance(arr, list):
        arr = np.array(arr, dtype=str)

    kwargs = kwargs or {}
    __validate_kwargs(kwargs)
    __validate_data(arr, False)
    remove_bad_chars = lambda x: str(x).replace('"', '').replace(',', '').replace('(', '').replace(')', '').replace("'", '')
    arr = [[remove_bad_chars(field) for field in row] for row in arr]
    arr = __preprocess_and_clean_in_memory(arr)

    arr = __transform(arr)

    output = __classify(arr, **kwargs)

    if remap:
        if kwargs.get('return_probabilities'):
            header = np.array([__get_key(i, mapping) for i in range(output.shape[1])], dtype=str).reshape(1, -1)
            output = np.concatenate((header, output), axis=0)
        else:
            output = np.array([__get_key(prediction, mapping) for prediction in output])

    return output


def validate(arr, labels):
    """
    Parameters
    ----------
    cleanarr : np.ndarray
        An array of float values that has undergone each pre-
        prediction step.

    Returns
    -------
    count : int
        A count of the number of instances in cleanarr.

    correct_count : int
        A count of the number of correctly classified instances in
        cleanarr.

    numeachclass : dict
        A dictionary mapping each class to its number of instances.

    outputs : np.ndarray
        The output of the predictor's '__classify' method on cleanarr.
    """
    predictions = predict(arr)
    correct_count = int(np.sum(predictions.reshape(-1) == labels.reshape(-1)))
    count = predictions.shape[0]
    
    class_0, class_1 = __get_key(0, mapping), __get_key(1, mapping)
    num_TP, num_TN, num_FP, num_FN, num_class_1, num_class_0 = 0, 0, 0, 0, 0, 0
    num_TP = int(np.sum(np.logical_and(predictions.reshape(-1) == class_1, labels.reshape(-1) == class_1)))
    num_TN = int(np.sum(np.logical_and(predictions.reshape(-1) == class_0, labels.reshape(-1) == class_0)))
    num_FN = int(np.sum(np.logical_and(predictions.reshape(-1) == class_0, labels.reshape(-1) == class_1)))
    num_FP = int(np.sum(np.logical_and(predictions.reshape(-1) == class_1, labels.reshape(-1) == class_0)))
    num_class_0 = int(np.sum(labels.reshape(-1) == class_0))
    num_class_1 = int(np.sum(labels.reshape(-1) == class_1))
    return count, correct_count, num_TP, num_TN, num_FP, num_FN, num_class_1, num_class_0, predictions


def __main():
    parser = argparse.ArgumentParser(description='Predictor trained on ' + str(TRAINFILE))
    parser.add_argument('csvfile', type=str, help='CSV file containing test set (unlabeled).')
    parser.add_argument('-validate', action='store_true', help='Validation mode. csvfile must be labeled. Output is classification statistics rather than predictions.')
    parser.add_argument('-headerless', help='Do not treat the first line of csvfile as a header.', action='store_true')
    parser.add_argument('-json', action="store_true", default=False, help="report measurements as json")
    parser.add_argument('-trim', action="store_true", help="If true, the prediction will not output ignored columns.")
    args = parser.parse_args()
    faulthandler.enable()

    arr, data, labels, header = load_data(csvfile=args.csvfile, headerless=args.headerless, validate=args.validate)

    if not args.validate:
        __write_predictions(arr, header, args.headerless, args.trim)
    else:

        count, correct_count, num_TP, num_TN, num_FP, num_FN, num_class_1, num_class_0, preds = validate(data, labels)

        classcounts = np.bincount(np.array([mapping[label.strip()] for label in labels], dtype='int32')).reshape(-1)
        class_balance = (classcounts[np.argwhere(classcounts > 0)] / arr.shape[0]).reshape(-1).tolist()
        best_guess = round(100.0 * np.max(class_balance), 2)
        H = float(-1.0 * sum([class_balance[i] * math.log(class_balance[i]) / math.log(2) for i in range(len(class_balance))]))
        modelacc = int(float(correct_count * 10000) / count) / 100.0
        mtrx, stats = __confusion_matrix(np.array(labels).reshape(-1), np.array(preds).reshape(-1), args.json)

        if args.json:
            json_dict = {'instance_count': count,
                         'classifier_type': classifier_type,
                         'classes': n_classes,
                         'number_correct': correct_count,
                         'accuracy': {
                             'best_guess': (best_guess/100),
                             'improvement': (modelacc - best_guess)/100,
                              'model_accuracy': (modelacc/100),
                         },
                         'model_capacity': model_cap,
                         'generalization_ratio': int(float(correct_count * 100) / model_cap) / 100.0 * H,
                         'model_efficiency': int(100 * (modelacc - best_guess) / model_cap) / 100.0,
                         'shannon_entropy_of_labels': H,
                         'class_balance': class_balance,
                         'confusion_matrix': mtrx.tolist(),
                         'multiclass_stats': stats}

            print(json.dumps(json_dict))
        else:
            pad = lambda s, length, pad_right: str(s) + ' ' * max(0, length - len(str(s))) if pad_right else ' ' * max(0, length - len(str(s))) + str(s)
            labels = np.array(list(mapping.keys())).reshape(-1, 1)
            max_class_name_len = max([len(clss) for clss in mapping.keys()] + [7])

            max_TP_len = max([len(str(stats[key]['TP'])) for key in stats.keys()] + [2])
            max_FP_len = max([len(str(stats[key]['FP'])) for key in stats.keys()] + [2])
            max_TN_len = max([len(str(stats[key]['TN'])) for key in stats.keys()] + [2])
            max_FN_len = max([len(str(stats[key]['FN'])) for key in stats.keys()] + [2])

            cmat_template_1 = "    {} | {}"
            cmat_template_2 = "    {} | " + " {} " * n_classes
            acc_by_class_template_1 = "    {} | {}  {}  {}  {}  {}  {}  {}  {}  {}  {}"

            acc_by_class_lengths = [max_class_name_len, max_TP_len, max_FP_len, max_TN_len, max_FN_len, 7, 7, 7, 7, 7, 7]
            acc_by_class_header_fields = ['target', 'TP', 'FP', 'TN', 'FN', 'TPR', 'TNR', 'PPV', 'NPV', 'F1', 'TS']
            print("Classifier Type:                    Neural Network")

            print(f"System Type:                        {n_classes}-way classifier\n")

            print("Accuracy:")
            print("    Best-guess accuracy:            {:.2f}%".format(best_guess))
            print("    Model accuracy:                 {:.2f}%".format(modelacc) + " (" + str(int(correct_count)) + "/" + str(count) + " correct)")
            print("    Improvement over best guess:    {:.2f}%".format(modelacc - best_guess) + " (of possible " + str(round(100 - best_guess, 2)) + "%)\n")

            print("Model capacity (MEC):               {:.0f} bits".format(model_cap))
            print("Generalization ratio:               {:.2f}".format(int(float(correct_count * 100) / model_cap) / 100.0 * H) + " bits/bit")

            if report_cmat:
                max_cmat_entry_len = len(str(int(np.max(mtrx))))
                mtrx = np.concatenate((labels, mtrx.astype('str')), axis=1).astype('str')
                max_pred_len = (mtrx.shape[1] - 1) * max_cmat_entry_len + n_classes * 2 - 1
                print("\nConfusion Matrix:\n")
                print(cmat_template_1.format(pad("Actual", max_class_name_len, False), "Predicted"))
                print(cmat_template_1.format("-" * max_class_name_len, "-" * max(max_pred_len, 9)))
                for row in mtrx:
                    print(cmat_template_2.format(
                        *[pad(field, max_class_name_len if i == 0 else max_cmat_entry_len, False) for i, field in enumerate(row)]))

            print("\nAccuracy by Class:\n")
            print(acc_by_class_template_1.format(
                *[pad(header_field, length, False) for i, (header_field, length) in enumerate(zip(acc_by_class_header_fields, acc_by_class_lengths))]))
            print(acc_by_class_template_1.format(
                *["-" * length for length in acc_by_class_lengths]))

            pct_format_string = "{:8.2%}"      # width = 8, decimals = 2
            for raw_class in mapping.keys():
                class_stats = stats[int(mapping[raw_class.strip()])]
                TP, FP, TN, FN = class_stats.get('TP', None), class_stats.get('FP', None), class_stats.get('TN', None), class_stats.get('FN', None)
                TPR = pct_format_string.format(class_stats['TPR']) if class_stats['TPR'] is not None else 'N/A'
                TNR = pct_format_string.format(class_stats['TNR']) if class_stats['TNR'] is not None else 'N/A'
                PPV = pct_format_string.format(class_stats['PPV']) if class_stats['PPV'] is not None else 'N/A'
                NPV = pct_format_string.format(class_stats['NPV']) if class_stats['NPV'] is not None else 'N/A'
                F1 = pct_format_string.format(class_stats['F1']) if class_stats['F1'] is not None else 'N/A'
                TS = pct_format_string.format(class_stats['TS']) if class_stats['TS'] is not None else 'N/A'
                line_fields = [raw_class, TP, FP, TN, FN, TPR, TNR, PPV, NPV, F1, TS]
                print(acc_by_class_template_1.format(
                    *[pad(field, length, False) for i, (field, length) in enumerate(zip(line_fields, acc_by_class_lengths))]))


if __name__ == "__main__":
    try:
        __main()
    except PredictorError as e:
        print(e, file=sys.stderr)
        sys.exit(e.code)
    except Exception as e:
        print(f"An unknown exception of type {type(e).__name__} occurred.", file=sys.stderr)
        sys.exit(-1)

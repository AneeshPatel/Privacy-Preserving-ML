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
# Invocation: brainome digits_clean_synth_labelled.csv -o digits_clean_synth_labelled.py -e 5 -y
# Total compiler execution time: 1:15:00.05. Finished on: May-02-2024 10:47:04.
# This source code requires Python 3.
#
"""

[01;1mPredictor:[0m                        
    Classifier Type:              Neural Network
    System Type:                  10-way classifier
    Training / Validation Split:  90% : 10%
    Accuracy:
      Best-guess accuracy:        77.72%
      Training accuracy:          81.42% (146564/179997 correct)
      Validation Accuracy:        81.32% (16267/20003 correct)
      Combined Model Accuracy:    81.41% (162831/200000 correct)


    Model Capacity (MEC):       535    bits
    Generalization Ratio:       346.37 bits/bit
    Percent of Data Memorized:     0.95%
    Resilience to Noise:          -2.44 dB




    Training Confusion Matrix:
              Actual | Predicted
              ------ | ---------
                 1.0 |  136552    3187       0       0       0       0       0     166       0       0 
                 0.0 |    8349    8021       0       0       0       0       0     295       0       0 
                 6.0 |    4329    2465       0      96       0       0       0     635       0       0 
                 8.0 |    5151    3240       0     560       0      83       0     118       0       0 
                 9.0 |     159    1594       0     283       0       1       0     338       0       0 
                 2.0 |       0     624       0      93       0     461       0     570       0       0 
                 3.0 |       0     991       0       8       0       0       0     402       0       0 
                 4.0 |       0       0       0       0       0     159       0     970       0       0 
                 5.0 |       0       0       0       0       0       0       0      79       0       0 
                 7.0 |       0       0       0       0       0       0       0      18       0       0 

    Validation Confusion Matrix:
              Actual | Predicted
              ------ | ---------
                 1.0 |   15170     357       0       0       0       0       0      18       0       0 
                 0.0 |     932     882       0       0       0       0       0      38       0       0 
                 6.0 |     477     282       0      10       0       0       0      68       0       0 
                 8.0 |     586     342       0      57       0      14       0      18       0       0 
                 9.0 |      17     189       0      26       0       0       0      32       0       0 
                 2.0 |       0      59       0      11       0      48       0      77       0       0 
                 3.0 |       0     107       0       2       0       0       0      47       0       0 
                 4.0 |       0       0       0       0       0      16       0     110       0       0 
                 5.0 |       0       0       0       0       0       0       0       9       0       0 
                 7.0 |       0       0       0       0       0       0       0       2       0       0 

    Training Accuracy by Class:
          Prediction |      TP      FP      TN      FN     TPR      TNR      PPV      NPV       F1       TS 
          ---------- | ------- ------- ------- ------- -------- -------- -------- -------- -------- --------
                 1.0 |  136552   17988   22104    3353   97.60%   55.13%   88.36%   86.83%   92.75%   86.48%
                 0.0 |    8021   12101  151231    8644   48.13%   92.59%   39.86%   94.59%   43.61%   27.88%
                 6.0 |       0       0  172472    7525    0.00%  100.00%      N/A   95.82%    0.00%    0.00%
                 8.0 |     560     480  170365    8592    6.12%   99.72%   53.85%   95.20%   10.99%    5.81%
                 9.0 |       0       0  177622    2375    0.00%  100.00%      N/A   98.68%    0.00%    0.00%
                 2.0 |     461     243  178006    1287   26.37%   99.86%   65.48%   99.28%   37.60%   23.15%
                 3.0 |       0       0  178596    1401    0.00%  100.00%      N/A   99.22%    0.00%    0.00%
                 4.0 |     970    2621  176247     159   85.92%   98.53%   27.01%   99.91%   41.10%   25.87%
                 5.0 |       0       0  179918      79    0.00%  100.00%      N/A   99.96%    0.00%    0.00%
                 7.0 |       0       0  179979      18    0.00%  100.00%      N/A   99.99%    0.00%    0.00%

    Validation Accuracy by Class:
          Prediction |      TP      FP      TN      FN     TPR      TNR      PPV      NPV       F1       TS 
          ---------- | ------- ------- ------- ------- -------- -------- -------- -------- -------- --------
                 1.0 |   15170    2012    2446     375   97.59%   54.87%   88.29%   86.71%   92.71%   86.40%
                 0.0 |     882    1336   16815     970   47.62%   92.64%   39.77%   94.55%   43.34%   27.67%
                 6.0 |       0       0   19166     837    0.00%  100.00%      N/A   95.82%    0.00%    0.00%
                 8.0 |      57      49   18937     960    5.60%   99.74%   53.77%   95.18%   10.15%    5.35%
                 9.0 |       0       0   19739     264    0.00%  100.00%      N/A   98.68%    0.00%    0.00%
                 2.0 |      48      30   19778     147   24.62%   99.85%   61.54%   99.26%   35.16%   21.33%
                 3.0 |       0       0   19847     156    0.00%  100.00%      N/A   99.22%    0.00%    0.00%
                 4.0 |     110     309   19568      16   87.30%   98.45%   26.25%   99.92%   40.37%   25.29%
                 5.0 |       0       0   19994       9    0.00%  100.00%      N/A   99.96%    0.00%    0.00%
                 7.0 |       0       0   20001       2    0.00%  100.00%      N/A   99.99%    0.00%    0.00%



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
TRAINFILE = ['digits_clean_synth_labelled.csv']
mapping = {'1.0': 0, '0.0': 1, '6.0': 2, '8.0': 3, '9.0': 4, '2.0': 5, '3.0': 6, '4.0': 7, '5.0': 8, '7.0': 9}
ignorelabels = []
ignorecolumns = []
target = 'Prediction'
target_column = 64
important_idxs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
expected_feature_cols = 64
classifier_type = 'NN'
num_attr = 64
n_classes = 10
model_cap = 535
w_h = np.array([[-0.21983212232589722, -0.16144047677516937, -0.10769282281398773, -0.49380433559417725, -0.29977720975875854, -0.4997551143169403, -0.28825148940086365, -0.48529133200645447, -0.46375951170921326, -0.5451157689094543, -0.5929304957389832, -0.11853239685297012, -0.38722652196884155, -0.35858047008514404, -0.49518880248069763, -0.147896409034729, -0.5169050693511963, -0.46809327602386475, -0.5774412155151367, -0.5949604511260986, -0.09243351221084595, -0.5749937891960144, -0.3906495273113251, -0.43466198444366455, -0.24403434991836548, -0.2727622389793396, -0.5733644366264343, -0.08944480121135712, -0.6400924324989319, -0.3217637538909912, -0.3748020529747009, -0.4178822636604309, 0.19810304045677185, -0.10507224500179291, -0.31264805793762207, -0.45319104194641113, -0.13439440727233887, -0.16787725687026978, -0.2983035147190094, -0.07951001822948456, -0.3382597267627716, -0.09141586720943451, -0.3189908564090729, -0.4078027606010437, -0.5966392755508423, -0.5062972903251648, -0.1405615657567978, -0.3607521951198578, -0.36987438797950745, -0.23732838034629822, -0.5417758822441101, -0.3701534569263458, -0.4322766065597534, -0.19841507077217102, -0.11393649876117706, -0.1679825335741043, -0.1337813436985016, -0.6103157997131348, -0.45371946692466736, -0.31609219312667847, -0.5974240303039551, -0.2716565430164337, -0.5410649180412292, -0.22594687342643738], [0.23527947068214417, -0.5991855263710022, -0.10692498832941055, -0.21487422287464142, -0.5423554182052612, -0.07226324826478958, -0.6022884249687195, -0.5462211966514587, -0.4951160252094269, -0.36523598432540894, 0.402709424495697, -0.43582770228385925, 0.07497531920671463, 0.31694963574409485, -0.12995545566082, -0.12662680447101593, -0.20499010384082794, -0.3083089292049408, -0.27828165888786316, 0.1264108121395111, -0.23279869556427002, -0.6398314833641052, 0.23046450316905975, -0.426908016204834, -0.13701210916042328, -0.3589896261692047, 0.21809852123260498, 0.4000650942325592, -0.5917292833328247, -0.427480548620224, -0.27000516653060913, 0.33124226331710815, -0.030529320240020752, -0.17284637689590454, -0.5713078379631042, -0.4972849488258362, -0.06912507116794586, -0.2926693558692932, -0.39758166670799255, 0.1505790650844574, -0.17563170194625854, -0.3990914821624756, 0.22942444682121277, -0.3980328440666199, -0.24495919048786163, 0.3134033977985382, -0.1813274323940277, 0.11339765787124634, -0.49314236640930176, -0.1289025843143463, -0.3123787045478821, -0.29197636246681213, -0.2549052834510803, -0.4192062318325043, -0.46676909923553467, -0.26711809635162354, -0.42753738164901733, -0.4705996513366699, -0.3121087849140167, -0.28020039200782776, -0.37676292657852173, -0.5471152663230896, 0.3065599501132965, -0.5152633190155029], [0.15300077199935913, -0.5431341528892517, -0.11872951686382294, -0.23473316431045532, -0.5994479656219482, -0.13478651642799377, -0.6380005478858948, -0.36741751432418823, -0.532016932964325, -0.6005986332893372, -0.3217132091522217, -0.49146464467048645, -0.14038684964179993, -0.579038679599762, -0.10922753810882568, -0.17513690888881683, -0.09590256959199905, -0.43564242124557495, -0.528702437877655, -0.5436474084854126, -0.10877759754657745, -0.4951033294200897, -0.38630905747413635, -0.39532342553138733, -0.30050423741340637, -0.5890589952468872, -0.11611232906579971, -0.5656166076660156, -0.06459423899650574, -0.4642624855041504, -0.48029112815856934, -0.39047321677207947, 0.08603471517562866, -0.4043782353401184, -0.6406494975090027, -0.1460416465997696, -0.06258007138967514, -0.2038489580154419, -0.06605743616819382, 0.07923063635826111, -0.37329497933387756, -0.5820577144622803, -0.6045898795127869, -0.5506625175476074, -0.4242733418941498, -0.18123868107795715, -0.47045430541038513, -0.3618064522743225, -0.35461848974227905, -0.540540874004364, -0.3583851456642151, -0.06982862204313278, -0.11817187070846558, -0.12911243736743927, -0.37918663024902344, -0.3453601896762848, -0.1915934979915619, -0.4472106397151947, -0.4683285057544708, -0.10179544240236282, -0.33391931653022766, -0.5467990636825562, -0.10556978732347488, -0.30226603150367737], [0.15554171800613403, -0.6029925346374512, -0.29656994342803955, -0.17270858585834503, -0.3924538493156433, -0.5157516002655029, -0.11793764680624008, -0.21530483663082123, -0.2677224576473236, -0.2817659378051758, -0.34409958124160767, -0.18380855023860931, -0.41406530141830444, -0.5216353535652161, -0.23951226472854614, -0.27295807003974915, -0.2398318350315094, -0.382804274559021, -0.3890931010246277, -0.3736204206943512, -0.2738629877567291, -0.5735876560211182, -0.6214663982391357, -0.24173679947853088, -0.5096754431724548, -0.37497037649154663, -0.26992514729499817, -0.309742271900177, -0.16311205923557281, -0.31240972876548767, -0.27854540944099426, -0.37653663754463196, 0.04291030764579773, -0.24392293393611908, -0.152195543050766, -0.6259584426879883, -0.47161105275154114, -0.6113729476928711, -0.4664165675640106, 0.17720535397529602, -0.6259555220603943, -0.6109891533851624, -0.18219338357448578, -0.5221854448318481, -0.13805243372917175, -0.2599365711212158, -0.2775450646877289, -0.1195794865489006, -0.6197907328605652, -0.13763700425624847, -0.25624221563339233, -0.5453172922134399, -0.27978479862213135, -0.4315340220928192, -0.467200368642807, -0.3819997012615204, -0.48757097125053406, -0.09631633013486862, -0.10030904412269592, -0.3815270960330963, -0.6083027124404907, -0.36829957365989685, -0.33130908012390137, -0.15620529651641846], [0.009353041648864746, -0.26590844988822937, -0.27338263392448425, -0.5154600143432617, -0.5308337211608887, -0.6103357076644897, -0.35279858112335205, -0.2604771852493286, -0.37118586897850037, -0.0683424174785614, -0.40597066283226013, -0.5402060151100159, -0.30206528306007385, -0.43238040804862976, -0.3903678059577942, -0.06884129345417023, -0.35885658860206604, -0.09657648950815201, -0.3116939961910248, -0.4586629271507263, -0.1373857706785202, -0.5161074995994568, -0.34089741110801697, -0.28517380356788635, -0.22003917396068573, -0.618733823299408, -0.3362553119659424, -0.42578405141830444, -0.16203522682189941, -0.25271129608154297, -0.23400378227233887, -0.1757289171218872, -0.06508241593837738, -0.3691409230232239, -0.11730001866817474, -0.5848175883293152, -0.5514540076255798, -0.3292495608329773, -0.4656704366207123, 0.01664873957633972, -0.09372226893901825, -0.3842148780822754, -0.5983774662017822, -0.4826618731021881, -0.08354586362838745, -0.41437968611717224, -0.47798171639442444, -0.21966364979743958, -0.31452813744544983, -0.6068208813667297, -0.10842405259609222, -0.1868150234222412, -0.09784300625324249, -0.22330540418624878, -0.4485267400741577, -0.44541433453559875, -0.20963560044765472, -0.4068692624568939, -0.29352283477783203, -0.6322016716003418, -0.4942156970500946, -0.24253572523593903, -0.3648470342159271, -0.4902704656124115], [-0.25356897711753845, 0.18510384857654572, 0.19237616658210754, 0.19039694964885712, 0.18851691484451294, 0.19120392203330994, 0.1901751309633255, 0.19477586448192596, 0.20593449473381042, 0.18639259040355682, 0.19112353026866913, 0.18665190041065216, 0.18562671542167664, 0.18978893756866455, 0.193954199552536, 0.18906007707118988, 0.19592450559139252, 0.19014357030391693, 0.18329112231731415, 0.19128233194351196, 0.194296732544899, 0.19763801991939545, 0.18554368615150452, 0.19767887890338898, 0.17818237841129303, 0.18695814907550812, 0.18549534678459167, 0.18892595171928406, 0.1886790245771408, 0.1930893212556839, 0.1922679990530014, 0.11912749707698822, 0.137405127286911, 0.18982012569904327, 0.19016793370246887, 0.19254344701766968, 0.19746270775794983, 0.19009365141391754, 0.18706370890140533, 0.05140179395675659, 0.17673732340335846, 0.18891873955726624, 0.18993936479091644, 0.19326151907444, 0.18581412732601166, 0.1978774219751358, 0.18298496305942535, 0.19213151931762695, 0.18978509306907654, 0.19014762341976166, 0.19075354933738708, 0.19258826971054077, 0.19981759786605835, 0.19346456229686737, 0.17662420868873596, 0.19319239258766174, 0.12114914506673813, 0.18597108125686646, 0.18885084986686707, 0.18698376417160034, 0.1914198249578476, 0.18886849284172058, 0.1927521824836731, 0.19606715440750122], [-0.21688756346702576, -0.3958696722984314, -0.07390108704566956, -0.2872624695301056, -0.3103487193584442, -0.3867116868495941, -0.16886018216609955, -0.49460193514823914, -0.1534174233675003, -0.28183943033218384, -0.31819474697113037, -0.310496062040329, -0.5297984480857849, -0.1346103549003601, -0.19783750176429749, -0.5778780579566956, -0.6211821436882019, -0.4253506362438202, -0.1573880910873413, -0.4936370551586151, -0.08108353614807129, -0.4966731667518616, -0.6131067276000977, -0.22264021635055542, -0.5261991620063782, -0.39634138345718384, -0.4411865472793579, -0.37361466884613037, -0.60382479429245, -0.6171470880508423, -0.22555026412010193, -0.38345664739608765, -0.17535966634750366, -0.3579213619232178, -0.6158480644226074, -0.2733514606952667, -0.21246111392974854, -0.06858935207128525, -0.5873420238494873, -0.03524959087371826, -0.19275397062301636, -0.08409994095563889, -0.5573413372039795, -0.22697892785072327, -0.23049592971801758, -0.1752607524394989, -0.5509297251701355, -0.1430669128894806, -0.4771345555782318, -0.5294565558433533, -0.2927950620651245, -0.10425151884555817, -0.5430242419242859, -0.19530825316905975, -0.38380852341651917, -0.257713258266449, -0.12374864518642426, -0.2715846300125122, -0.5918431282043457, -0.4380388557910919, -0.2928183078765869, -0.5957484245300293, -0.3021868169307709, -0.5066131353378296]])
b_h = np.array([-0.12227106094360352, -0.09522780030965805, -0.1831778734922409, -0.18396756052970886, -0.4106088876724243, -66.65166473388672, -0.28406718373298645])
w_o = np.array([[0.23242466151714325, 0.5849385857582092, -0.14949922263622284, 0.34319210052490234, -0.03979194536805153, 0.8105828762054443, -0.23937994241714478], [-0.1379859447479248, -0.7192456126213074, -0.1769929677248001, -0.07281909883022308, -0.7638978958129883, 0.18352903425693512, 0.130912646651268], [-0.8286478519439697, -0.2705118954181671, -0.7662268877029419, -0.7625184655189514, 0.04423676058650017, 0.10184438526630402, 0.21106630563735962], [-0.7319279313087463, -0.09469705820083618, -0.6034596562385559, 0.3787190914154053, -0.5428481101989746, 0.061125509440898895, -0.23165468871593475], [-0.07954953610897064, 0.15903963148593903, -0.7630506157875061, 0.2040034830570221, -0.38874778151512146, -0.1917429268360138, -0.5175190567970276], [-0.9380924701690674, -0.7975778579711914, -0.8716782331466675, 0.07361489534378052, -0.1761920154094696, -0.7279309630393982, 0.028351768851280212], [0.05832898989319801, -0.6458799242973328, -0.3041418790817261, -0.23829945921897888, -0.7151600122451782, -0.6152763962745667, 0.22888651490211487], [-0.8086051344871521, -0.1774471551179886, -0.002108997432515025, -0.4399493932723999, -0.2975635230541229, -6.175447940826416, 0.02188176102936268], [-0.36008140444755554, -0.369575560092926, -0.6306204795837402, -0.3449776768684387, 0.0028555230237543583, -38.309783935546875, -0.539579451084137], [-0.2285587638616562, -0.7514145374298096, -0.2911081612110138, -0.5466809272766113, -0.6978701949119568, -52.26799011230469, 0.06581547111272812]])
b_o = np.array([-2.196413516998291, 1.6370165348052979, 1.3415864706039429, 1.7699037790298462, 1.4750603437423706, 2.1867663860321045, 1.8469449281692505, 2.4890387058258057, -0.1610538810491562, -1.5596355199813843])


class PredictorError(Exception):

    def __init__(self, msg, code):
        self.msg = msg
        self.code = code

    def __str__(self):
        return self.msg

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
        max_vals = np.tile(np.max(out, axis=1).reshape(-1, 1), out.shape[1])
        exps = np.exp(out - max_vals)
        Z = np.sum(exps, axis=1).reshape(-1, 1)
        output = exps / Z
    else:
        output = np.argmax(out, axis=1).reshape(-1)
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
    
    numeachclass = {clss : np.argwhere(labels == clss).shape[0] for clss in np.unique(labels)}
    return count, correct_count, numeachclass, predictions


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

        count, correct_count, numeachclass, preds = validate(data, labels)

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

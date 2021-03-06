from sklearn.feature_selection import SelectFromModel, RFE
import json
from feature_selection.dataset import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
import time

np.random.seed(123)

X = datasets[1].X_train
ss = StandardScaler().fit(X)
X = ss.transform(X)
y = datasets[1].y_train
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state=123, test_size=0.2)


class SelectedFeatures:
    def __init__(self, dataset_name, features, method):
        self.dataset_name = dataset_name
        self.features = features
        self.method = method


def target_correlation(X_train, y_train, threshold):
    df = pd.DataFrame(np.hstack((X_train, y_train.reshape(-1, 1))))
    corr_matrix = np.abs(df.corr())
    corr = [[i, corr_matrix[corr_matrix.columns[-1]][i]] for i in corr_matrix[corr_matrix.columns[-1]][:-1].index]
    corr = sorted(corr, key=lambda x: -x[1])
    features = [corr[0][0]]
    for i, row in enumerate(corr):
        if np.max(corr_matrix[row[0]][features]) < .9 and row[1] > threshold:
            features.append(row[0])
    return sorted(features)


def rf_importance(X_train, y_train, threshold):
    clf = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
    sfm = SelectFromModel(clf, threshold=threshold)
    sfm.fit(X_train, y_train.reshape(-1))
    return sfm.get_support(indices=True)


def rfe_selection(X_train, y_train, n_features):
    rfe_selector = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=n_features, step=1)
    rfe_selector.fit(X_train, y_train.reshape(-1))
    return rfe_selector.get_support(indices=True)


# print(target_correlation(datasets[0], 0.1))
# # [48, 128, 336, 338, 442, 475]
# print(rf_importance(datasets[0], 0.01))
# # [105 241 338 442 475]
# print(rfe_selection(datasets[0], 6))
# # [ 28  48 105 338 442 475]

path = "G:/Biblioteki/Desktop/AML_results/results2.json"
results = {}
results["dataset"] = "artificial"
start = time.time()
features = target_correlation(X_train, y_train, 0.1)
time_delta = time.time() - start
results["method"] = "target_correlation"
results['time'] = time_delta
classifiers = [AdaBoostClassifier, GradientBoostingClassifier, DecisionTreeClassifier, RandomForestClassifier,
                               QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis, LogisticRegression]
for classifier in classifiers:
    cl = classifier().fit(X_train[:, features], y_train)
    results[classifier.__name__] = np.sum(cl.predict(X_test[:, features]) == y_test) / y_test.shape[0]
results["features"] = list([int(i) for i in features])
print(results)
with open(path, "w+") as f:
    json.dump(results, fp=f, indent=4)
    f.close()

start = time.time()
features = rf_importance(X_train, y_train, 0.01)
time_delta = time.time() - start
results["method"] = "rf_importance"
results['time'] = time_delta
classifiers = [AdaBoostClassifier, GradientBoostingClassifier, DecisionTreeClassifier, RandomForestClassifier,
                               QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis, LogisticRegression]
for classifier in classifiers:
    cl = classifier().fit(X_train[:, features], y_train)
    results[classifier.__name__] = np.sum(cl.predict(X_test[:, features]) == y_test) / y_test.shape[0]
results["features"] = list([int(i) for i in features])
print(results)
with open(path, "a") as f:
    json.dump(results, fp=f, indent=4)
    f.close()

start = time.time()
features = rfe_selection(X_train, y_train, 6)
time_delta = time.time() - start
results["method"] = "rfe_selection"
results['time'] = time_delta
classifiers = [AdaBoostClassifier, GradientBoostingClassifier, DecisionTreeClassifier, RandomForestClassifier,
                               QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis, LogisticRegression]
for classifier in classifiers:
    cl = classifier().fit(X_train[:, features], y_train)
    results[classifier.__name__] = np.sum(cl.predict(X_test[:, features]) == y_test) / y_test.shape[0]
results["features"] = list([int(i) for i in features])
print(results)
with open(path, "a") as f:
    json.dump(results, fp=f, indent=4)
    f.close()

# print(target_correlation(datasets[1], 0.1))
# # [2, 12, 16, 17, 34, 38, 43, 48, 53, 60, 61, 66, 68, 70, 82, 83, 95, 96, 101, 106, 112, 114, 131, 139, 145, 155, 162, 164, 166, 176, 189, 195, 205, 212, 214, 216, 225, 229, 238, 239, 246, 252, 257, 266, 267, 274, 286, 288, 291, 295, 300, 312, 313, 315, 319, 323, 328, 329, 348, 352, 356, 362, 364, 365, 367, 378, 379, 391, 393, 398, 401, 403, 409, 415, 426, 435, 439, 443, 451, 455, 457, 458, 461, 462, 467, 470, 471, 483, 487, 493, 495, 497, 499, 502, 507, 509, 510, 511, 514, 518, 521, 526, 532, 537, 543, 554, 557, 561, 568, 576, 579, 582, 583, 592, 593, 596, 600, 616, 629, 631, 637, 641, 642, 649, 651, 653, 657, 661, 665, 666, 668, 673, 678, 681, 682, 695, 698, 707, 709, 721, 723, 729, 742, 756, 757, 761, 764, 766, 767, 776, 782, 792, 796, 803, 813, 818, 827, 828, 834, 840, 841, 846, 851, 852, 855, 858, 864, 865, 875, 879, 880, 884, 893, 902, 911, 916, 920, 934, 948, 961, 967, 976, 981, 986, 1030, 1032, 1042, 1049, 1055, 1067, 1069, 1071, 1073, 1084, 1087, 1095, 1104, 1108, 1109, 1125, 1128, 1133, 1134, 1149, 1155, 1164, 1170, 1175, 1180, 1184, 1185, 1186, 1212, 1218, 1220, 1222, 1225, 1228, 1236, 1240, 1243, 1248, 1258, 1265, 1271, 1275, 1277, 1279, 1282, 1293, 1295, 1301, 1313, 1328, 1332, 1340, 1344, 1349, 1358, 1359, 1364, 1376, 1387, 1388, 1389, 1392, 1393, 1400, 1402, 1406, 1407, 1424, 1438, 1458, 1479, 1482, 1485, 1488, 1491, 1495, 1500, 1504, 1506, 1511, 1522, 1532, 1536, 1539, 1546, 1547, 1555, 1556, 1558, 1564, 1565, 1567, 1575, 1576, 1586, 1588, 1592, 1597, 1599, 1600, 1606, 1614, 1618, 1628, 1638, 1642, 1645, 1652, 1654, 1656, 1660, 1663, 1673, 1682, 1683, 1686, 1687, 1693, 1697, 1702, 1703, 1708, 1709, 1716, 1718, 1725, 1732, 1742, 1743, 1745, 1746, 1750, 1753, 1755, 1761, 1762, 1771, 1776, 1783, 1787, 1791, 1793, 1794, 1796, 1804, 1813, 1818, 1821, 1822, 1831, 1837, 1853, 1860, 1861, 1862, 1863, 1870, 1873, 1881, 1905, 1908, 1909, 1911, 1913, 1922, 1923, 1927, 1929, 1935, 1936, 1941, 1948, 1951, 1954, 1965, 1966, 1967, 1968, 1980, 1983, 1987, 1992, 1995, 2007, 2008, 2016, 2032, 2048, 2051, 2052, 2056, 2068, 2079, 2094, 2098, 2099, 2101, 2104, 2114, 2116, 2123, 2132, 2133, 2141, 2143, 2149, 2152, 2155, 2168, 2169, 2175, 2183, 2187, 2189, 2201, 2222, 2235, 2237, 2244, 2251, 2253, 2257, 2282, 2283, 2286, 2301, 2304, 2305, 2314, 2316, 2321, 2325, 2333, 2334, 2353, 2354, 2366, 2367, 2368, 2372, 2378, 2381, 2387, 2392, 2401, 2404, 2421, 2422, 2426, 2437, 2446, 2451, 2464, 2465, 2469, 2471, 2474, 2475, 2479, 2483, 2487, 2488, 2491, 2496, 2498, 2509, 2510, 2513, 2539, 2544, 2554, 2556, 2559, 2565, 2566, 2571, 2589, 2595, 2596, 2615, 2618, 2621, 2628, 2631, 2634, 2638, 2641, 2660, 2662, 2670, 2673, 2676, 2682, 2688, 2693, 2707, 2714, 2717, 2721, 2733, 2740, 2742, 2745, 2760, 2764, 2767, 2768, 2769, 2770, 2782, 2783, 2784, 2788, 2801, 2808, 2812, 2820, 2831, 2835, 2848, 2856, 2863, 2869, 2873, 2886, 2887, 2893, 2895, 2906, 2912, 2917, 2923, 2924, 2926, 2937, 2944, 2959, 2962, 2971, 2983, 2990, 2996, 3002, 3009, 3010, 3011, 3013, 3022, 3031, 3034, 3035, 3037, 3044, 3046, 3051, 3054, 3057, 3061, 3062, 3065, 3066, 3073, 3074, 3077, 3085, 3086, 3105, 3122, 3133, 3153, 3159, 3162, 3171, 3172, 3187, 3188, 3197, 3206, 3223, 3230, 3247, 3248, 3251, 3253, 3254, 3257, 3266, 3269, 3275, 3283, 3302, 3304, 3316, 3320, 3327, 3336, 3347, 3359, 3360, 3365, 3372, 3374, 3376, 3385, 3389, 3402, 3409, 3418, 3423, 3424, 3433, 3443, 3450, 3453, 3454, 3463, 3468, 3480, 3486, 3488, 3491, 3503, 3508, 3514, 3518, 3521, 3532, 3543, 3557, 3560, 3566, 3570, 3585, 3587, 3594, 3603, 3605, 3608, 3624, 3633, 3637, 3642, 3643, 3647, 3648, 3656, 3658, 3659, 3663, 3666, 3668, 3670, 3675, 3679, 3683, 3686, 3688, 3693, 3694, 3699, 3700, 3707, 3721, 3725, 3730, 3737, 3752, 3755, 3756, 3759, 3770, 3774, 3776, 3782, 3796, 3801, 3821, 3827, 3831, 3833, 3837, 3846, 3857, 3865, 3869, 3876, 3885, 3893, 3897, 3932, 3958, 3965, 3970, 3975, 4000, 4002, 4008, 4021, 4023, 4028, 4031, 4045, 4062, 4063, 4064, 4073, 4093, 4094, 4097, 4102, 4104, 4105, 4106, 4108, 4109, 4114, 4129, 4143, 4146, 4152, 4158, 4164, 4168, 4177, 4183, 4187, 4188, 4194, 4195, 4196, 4197, 4198, 4200, 4202, 4217, 4228, 4239, 4267, 4271, 4273, 4275, 4289, 4290, 4301, 4353, 4362, 4365, 4377, 4379, 4380, 4390, 4403, 4409, 4415, 4416, 4424, 4433, 4445, 4447, 4456, 4461, 4466, 4486, 4489, 4493, 4503, 4506, 4507, 4524, 4543, 4552, 4553, 4557, 4567, 4573, 4575, 4577, 4585, 4588, 4596, 4598, 4599, 4605, 4608, 4610, 4616, 4618, 4619, 4642, 4655, 4660, 4671, 4674, 4681, 4689, 4690, 4694, 4719, 4721, 4725, 4733, 4751, 4753, 4758, 4761, 4766, 4779, 4788, 4790, 4794, 4799, 4802, 4808, 4823, 4830, 4831, 4835, 4844, 4845, 4862, 4865, 4869, 4874, 4878, 4888, 4893, 4895, 4904, 4906, 4916, 4917, 4921, 4922, 4924, 4925, 4935, 4936, 4941, 4949, 4955, 4962, 4963, 4967, 4974, 4976, 4977, 4979, 4980, 4981, 4990, 4991, 4999]
# print(rf_importance(datasets[1], 0.01))
# # [ 511  557 3002 3656 3975 4507]
# print(rfe_selection(datasets[1], 6))
# # [ 338 1079 1375 1656 2007 3656]

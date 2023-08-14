import numpy as np
# import tensorflow as tf
# import tensorflow.compat.v1 as tf
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
tf.compat.v1.enable_eager_execution()
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import scipy.io as io
from gaminet import GAMINet
from gaminet.utils import local_visualize
from gaminet.utils import global_visualize_density
from gaminet.utils import feature_importance_visualize
from gaminet.utils import plot_trajectory
from gaminet.utils import plot_regularization
from gaminet.utils import save_dict_logs
from gaminet.utils import save_dict_global_importance
from gaminet.utils import save_dict_global_density
import scipy.io as sio

from matplotlib import pylab as plt
import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING) #DEBUG, INFO, WARNING, ERROR
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # 使用第二块GPU（从0开始）
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)  # 多块GPU按需使用

def metric_wrapper(metric, scaler):
    def wrapper(label, pred):
        return metric(label, pred, scaler=scaler)
    return wrapper

def rmse(label, pred, scaler):
    pred = scaler.inverse_transform(pred.reshape([-1, 1]))
    label = scaler.inverse_transform(label.reshape([-1, 1]))
    return np.sqrt(np.mean((pred - label) ** 2))

def data_generator1():
    task_type = "Regression"
    meta_info = {"X1": {"type": 'continuous'},
                 'X2': {'type': 'continuous'},
                 'X3': {'type': 'continuous'},
                 'X4': {'type': 'continuous'},
                 'X5': {'type': 'continuous'},
                 'X6': {'type': 'continuous'},
                 'X7': {'type': 'continuous'},
                 'X8': {'type': 'continuous'},
                 'X9': {'type': 'continuous'},
                 'X10': {'type': 'continuous'},
                 "X1t1": {"type": 'continuous'},
                 'X2t1': {'type': 'continuous'},
                 'X3t1': {'type': 'continuous'},
                 'X4t1': {'type': 'continuous'},
                 'X5t1': {'type': 'continuous'},
                 'X6t1': {'type': 'continuous'},
                 'X7t1': {'type': 'continuous'},
                 'X8t1': {'type': 'continuous'},
                 'X9t1': {'type': 'continuous'},
                 'X10t1': {'type': 'continuous'},
                 "X1t2": {"type": 'continuous'},
                 'X2t2': {'type': 'continuous'},
                 'X3t2': {'type': 'continuous'},
                 'X4t2': {'type': 'continuous'},
                 'X5t2': {'type': 'continuous'},
                 'X6t2': {'type': 'continuous'},
                 'X7t2': {'type': 'continuous'},
                 'X8t2': {'type': 'continuous'},
                 'X9t2': {'type': 'continuous'},
                 'X10t2': {'type': 'continuous'},
                 "X1t3": {"type": 'continuous'},
                 'X2t3': {'type': 'continuous'},
                 'X3t3': {'type': 'continuous'},
                 'X4t3': {'type': 'continuous'},
                 'X5t3': {'type': 'continuous'},
                 'X6t3': {'type': 'continuous'},
                 'X7t3': {'type': 'continuous'},
                 'X8t3': {'type': 'continuous'},
                 'X9t3': {'type': 'continuous'},
                 'X10t3': {'type': 'continuous'},   ####### ！！！！
                 'Y': {'type': 'target'}}
    NumData = io.loadmat('NumData.mat')   ##### ！！！！
    train_x = NumData['tr_xall']
    train_y = NumData['tr_y']
    test_x = NumData['ts_xall']
    test_y = NumData['ts_y']
    val_x = NumData['val_xall']
    val_y = NumData['val_y']

    for i, (key, item) in enumerate(meta_info.items()):
        if item['type'] == 'target':
            sy = MinMaxScaler((0, 1))
            train_y = sy.fit_transform(train_y)
            meta_info[key]['scaler'] = sy
        else:
            sx = MinMaxScaler((0, 1))
            sx.fit([[0], [1]])
            train_x[:, [i]] = sx.transform(train_x[:, [i]])
            meta_info[key]['scaler'] = sx
    return train_x, test_x, train_y, test_y, val_x, val_y, task_type, meta_info, metric_wrapper(rmse, sy)

train_x, test_x, train_y, test_y, val_x, val_y, task_type, meta_info, get_metric = data_generator1()

#### Model Training, Validating, Testing ####
model = GAMINet(meta_info=meta_info, interact_num=20, interact_arch=[20, 10],
            subnet_arch=[20, 10], task_type=task_type, activation_func=tf.tanh, main_grid_size=41, interact_grid_size=41,
            batch_size=min(500, int(0.2*train_x.shape[0])), lr_bp=0.001, main_effect_epochs=2000,
            interaction_epochs=2000, tuning_epochs=50, loss_threshold=0.01,
            verbose=True, val_ratio=0.2, early_stop_thres=100)  ##### ！！！！
model.fit(train_x, train_y, val_x, val_y)

#### Show Results ####
val_x = val_x
val_y = val_y
tr_x = train_x
tr_y =train_y
pred_train = model.predict(tr_x)
pred_val = model.predict(val_x)
pred_test = model.predict(test_x)
gaminet_stat = np.hstack([np.round(get_metric(tr_y, pred_train),5),
                      np.round(get_metric(val_y, pred_val),5),
                      np.round(get_metric(test_y, pred_test),5)])
print(gaminet_stat)

#测试集R2计算
SStot = np.sum(np.square(test_y - np.mean(test_y)))
SSres = np.sum(np.square(test_y - pred_test))
R2_ts = 1 - SSres/SStot
print("Testing R2:", "{:.9f}".format(R2_ts))

# simu_dir = "./resultsTest/Num/DynInteract/"    #### ！！！！
if not os.path.exists(simu_dir):
    os.makedirs(simu_dir)

# sio.savemat(simu_dir + 'Results.mat', {'pred_train': pred_train, 'pred_val': pred_val, 'pred_test': pred_test, 'tr_y': tr_y, 'val_y': val_y, 'test_y': test_y, 'gaminet_stat': gaminet_stat, 'R2_ts': R2_ts})
data_dict_logs = model.summary_logs(save_dict=False)
plot_trajectory(data_dict_logs, folder=simu_dir, name="s1_traj_plot", log_scale=True, save_png=True, save_eps=False)
plot_regularization(data_dict_logs, folder=simu_dir, name="s1_regu_plot", log_scale=True, save_png=True, save_eps=False)
save_dict_logs(data_dict_logs, folder=simu_dir)

data_dict_global = model.global_explain(save_dict=False)
feature_importance_visualize(data_dict_global, save_png=True, folder=simu_dir, name='s1_feature')
save_dict_global_importance(data_dict_global, folder=simu_dir)

global_visualize_density(data_dict_global, save_png=True, folder=simu_dir, name='s1_global')
save_dict_global_density(data_dict_global, folder=simu_dir)

# data_dict_local = model.local_explain(train_x[[0]], train_y[[0]], save_dict=False)
# local_visualize(data_dict_local, save_png=True, folder=simu_dir, name='s1_local')

# 画图
s = range(len(test_y))
fig,ax = plt.subplots()
plt.plot(s, test_y, 'b.-', linewidth=2, label='Real Value')
plt.plot(s, pred_test, 'ro-', linewidth=2, label='Predicted Value')
plt.title('Prediction Plots')
plt.xlabel('Sample Points')
plt.ylabel('Output')
plt.legend(loc= 'best')
plt.show()




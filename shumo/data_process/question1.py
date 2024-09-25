import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor


def get_files_by_keywords(directory, keywords):
    # 遍历指定文件夹，根据关键字数组提取文件。
    matched_files = []
    for filename in os.listdir(directory):
        if all(keyword in filename for keyword in keywords):
            matched_files.append(filename)
    return matched_files


def process_array_string(array_string):
    # 去掉最大值、最小值并计算平均值
    # 如果输入是单个值，则直接返回
    if isinstance(array_string, (int, float)):
        return array_string
    # 将字符串转换为列表
    values = ast.literal_eval(array_string)
    # 如果列表长度大于2，去掉最大值和最小值
    if len(values) > 2:
        values.remove(max(values))
        values.remove(min(values))
    mean_value = np.mean(values)
    return mean_value


def evaluation(random_state, y_test, y_pred, output, name):
    # 对比曲线图：真实值 vs 预测值
    plt.figure(random_state, figsize=(10, 6))
    plt.plot(np.arange(len(y_test)), y_test, label="true seq_time", color="blue", marker="o")
    plt.plot(np.arange(len(y_pred)), y_pred, label="pred seq_time", color="red", marker="x")
    plt.xlabel("Test Samples")
    plt.ylabel("seq_time")
    plt.title("true vs pred seq_time")
    plt.legend()
    plt.savefig(output + name)


def feature_importance(random_state, importance, X_train, output, name):
    #### 只有树模型才有
    # 打印每个特征的重要性
    # 可视化特征重要性
    plt.figure(random_state + 1, figsize=(10, 6))
    plt.barh(X_train.columns, importance)
    plt.xlabel("Feature Importance")
    plt.ylabel("Features")
    plt.title("Feature Importance Analysis")
    plt.savefig(output + name)


def preprocess_2(training_data_names, data_dir):
    #### 读取所有训练数据 #####
    training_data_all = pd.DataFrame()
    file_split_id = []  # 记录分隔文件的id位置
    for file in training_data_names:
        file_path = os.path.join(data_dir, file)
        df = pd.read_csv(file_path)
        # 获取当前合并DataFrame中的最大test_id，如果为空则设置为0
        if not training_data_all.empty:
            max_test_id = training_data_all["test_id"].max()
        else:
            max_test_id = 0
        # 调整新df的test_id，保证test_id连续递增
        df["test_id"] = df["test_id"] + max_test_id
        # 将当前DataFrame追加到总的training_data_all中
        training_data_all = pd.concat([training_data_all, df], ignore_index=True)
        file_split_id.append(training_data_all["test_id"].max())

    # columns_class = ["ap_id", "sta_id"]
    # protocol_name = ["tcp", "udp"]

    ### 提取对应的列rssi ###
    training_data_all_ap_0 = training_data_all.loc[training_data_all["ap_id"] == "ap_0"].copy()
    for i, column in enumerate(ap_0_sta_0):
        training_data_all_ap_0[column] = training_data_all_ap_0[column].apply(process_array_string)

    training_data_all_ap_1 = training_data_all.loc[training_data_all["ap_id"] == "ap_1"].copy()
    for i, column in enumerate(ap_1_sta_1):
        training_data_all_ap_1[ap_0_sta_0[i]] = training_data_all_ap_1[column].apply(process_array_string)

    training_data_all_processed = pd.concat(
        [training_data_all_ap_0[columns_basic + ap_0_sta_0], training_data_all_ap_1[columns_basic + ap_0_sta_0]],
        ignore_index=True)

    ######## 训练模型 #######
    training_data = training_data_all_processed.loc[:, columns_basic + ap_0_sta_0].copy()
    # 编码非数值变量
    training_data_encoded = pd.get_dummies(training_data, columns=["protocol"])
    protocol_num_list = [1, 2]
    protocol_name_list = ["tcp", "udp"]
    training_data["protocol"] = training_data["protocol"].map(dict(zip(protocol_name_list, protocol_num_list)))

    # 拼接向量
    X = training_data_encoded[columns_numerical + ap_0_sta_0 +
                              [col for col in training_data_encoded.columns if col.startswith("protocol_")]]
    y = training_data_encoded["seq_time"]

    return X, y


def preprocess_3(training_data_names, data_dir):
    #### 读取所有训练数据 #####
    training_data_all = pd.DataFrame()
    file_split_id = []  # 记录分隔文件的id位置
    for file in training_data_names:
        file_path = os.path.join(data_dir, file)
        df = pd.read_csv(file_path)
        # 获取当前合并DataFrame中的最大test_id，如果为空则设置为0
        if not training_data_all.empty:
            max_test_id = training_data_all["test_id"].max()
        else:
            max_test_id = 0
        # 调整新df的test_id，保证test_id连续递增
        df["test_id"] = df["test_id"] + max_test_id
        # 将当前DataFrame追加到总的training_data_all中
        training_data_all = pd.concat([training_data_all, df], ignore_index=True)
        file_split_id.append(training_data_all["test_id"].max())

    # columns_class = ["ap_id", "sta_id"]
    # protocol_name = ["tcp", "udp"]

    training_data_all_ap_0 = training_data_all.loc[training_data_all["ap_id"] == "ap_0"].copy()
    for i, column in enumerate(ap_0_sta_0):
        training_data_all_ap_0[column] = training_data_all_ap_0[column].apply(process_array_string)

    training_data_all_ap_1 = training_data_all.loc[training_data_all["ap_id"] == "ap_1"].copy()
    for i, column in enumerate(ap_1_sta_1):
        training_data_all_ap_1[ap_0_sta_0[i]] = training_data_all_ap_1[column].apply(process_array_string)

    training_data_all_ap_2 = training_data_all.loc[training_data_all["ap_id"] == "ap_2"].copy()
    for i, column in enumerate(ap_2_sta_2):
        training_data_all_ap_2[ap_0_sta_0[i]] = training_data_all_ap_2[column].apply(process_array_string)

    training_data_all_processed = pd.concat(
        [training_data_all_ap_0[columns_basic + ap_0_sta_0], training_data_all_ap_1[columns_basic + ap_0_sta_0]],
        ignore_index=True)
    training_data_all_processed = pd.concat([training_data_all_processed, training_data_all_ap_2[columns_basic + ap_0_sta_0]],
                                            ignore_index=True)

    ######## 训练模型 #######
    # 去掉空值
    training_data = training_data_all_processed.loc[:, columns_basic + ap_0_sta_0].dropna().copy()
    # 编码非数值变量
    training_data_encoded = pd.get_dummies(training_data, columns=["protocol"])

    # 拼接向量
    X = training_data_encoded[columns_numerical + ap_0_sta_0 +
                              [col for col in training_data_encoded.columns if col.startswith("protocol_")]]
    y = training_data_encoded["seq_time"]

    return X, y


def predict_test_2(data_dir, ap_name, output, scaler, X_train):
    test_data_names = get_files_by_keywords(data_dir, ["test", ap_name, "csv"])
    test_data_names = sorted(test_data_names)
    for test_data_name in test_data_names:
        file_path = os.path.join(data_dir, test_data_name)
        test_data_all = pd.read_csv(file_path)

        test_data_ap_0 = test_data_all.loc[test_data_all["ap_id"] == "ap_0"].copy()
        for i, column in enumerate(ap_0_sta_0):
            test_data_ap_0[column] = test_data_ap_0[column].apply(process_array_string)

        test_data_ap_1 = test_data_all.loc[test_data_all["ap_id"] == "ap_1"].copy()
        for i, column in enumerate(ap_1_sta_1):
            test_data_ap_1[ap_0_sta_0[i]] = test_data_ap_1[column].apply(process_array_string)

        test_data_processed = pd.concat(
            [test_data_ap_0[columns_basic + ap_0_sta_0], test_data_ap_1[columns_basic + ap_0_sta_0]], ignore_index=True)

        ####### 预测数据 ########
        test_data = test_data_processed.loc[:, columns_basic + ap_0_sta_0].copy()
        test_data_encoded = test_data
        # 如果测试集中缺少训练集中的某些特征列，则补 0
        for col in X_train.columns:
            if col not in test_data_encoded.columns:
                test_data_encoded[col] = 0  # 补充缺失的列并填充 0

        # 保证测试集的列顺序与训练集相同
        test_data_encoded = test_data_encoded[X_train.columns]
        # 对测试数据进行归一化（使用与训练集相同的 scaler）
        X_test_final_scaled = scaler.transform(test_data_encoded)

        # 使用训练好的模型进行预测
        y_test_pred = model.predict(X_test_final_scaled)
        # 输出预测结果
        test_data_all["predict seq_time"] = y_test_pred
        test_data_all.to_csv(f"{output}/{test_data_name}", index=False)
        test_data["predict seq_time"] = y_test_pred


def split_dataset(X, y, random_state):
    # 将数据分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    # 归一化特征
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler


def build_model(X_train_scaled, X_test_scaled, y_train, y_test, random_state):
    #### 模型训练 ####
    # 线性回归
    # model = LinearRegression()
    # XGBoost
    model = xgb.XGBRegressor(objective="reg:squarederror", random_state=random_state)
    # 随机森林
    # model = RandomForestRegressor(random_state=random_state)
    # 支持向量回归
    # model = SVR(kernel='rbf')
    # 神经网络回归模型
    # model = MLPRegressor(hidden_layer_sizes=(64, 32, 16, 32), max_iter=1000, alpha=0.001, random_state=random_state)

    # 训练模型
    model.fit(X_train_scaled, y_train)
    # 对测试集进行预测
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    return model, y_pred, mse


def model_predict(X, y, random_state=None):
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = split_dataset(X, y, random_state)
    model, y_pred, mse = build_model(X_train_scaled, X_test_scaled, y_train, y_test, random_state)
    return model, y_pred, mse, scaler, X_train


def loop_test(X, y, begin, max_loop, ap_name, output):
    min_ = 100
    for random_state in range(begin, max_loop):
        model, y_pred, mse, scaler, X_train = model_predict(X, y, random_state)
        if random_state % 200 == 0:
            print(random_state)
        if mse < min_:
            min_ = mse
            print(f"Mean Squared Error: {mse}, Random state: {random_state}")
            # evaluation(random_state, y_test, y_pred, output, 'q1_' + ap_name + '_evaluation.png')
            # feature_importance(random_state, model.feature_importances_, X_train, output, 'q1_' + ap_name + '_importance.png')


if __name__ == '__main__':
    # ##### 修改这里 #####
    ap_name = "2ap"
    data_dir = os.getcwd() + "/dataset/"
    output = './output/'
    training_data_names = get_files_by_keywords(data_dir, ["training", ap_name, "csv"])
    training_data_names = sorted(training_data_names)
    columns_numerical = ["eirp", "nav"]
    if ap_name == '2ap':
        ap_0_sta_0 = [
            "ap_from_ap_1_mean_ant_rssi", "sta_to_ap_0_mean_ant_rssi", "sta_to_ap_1_mean_ant_rssi",
            "sta_from_ap_0_mean_ant_rssi", "sta_from_ap_1_mean_ant_rssi", "sta_from_sta_1_rssi"
        ]
        ap_1_sta_1 = [
            "ap_from_ap_0_mean_ant_rssi", "sta_to_ap_1_mean_ant_rssi", "sta_to_ap_0_mean_ant_rssi",
            "sta_from_ap_1_mean_ant_rssi", "sta_from_ap_0_mean_ant_rssi", "sta_from_sta_0_rssi"
        ]
        columns_basic = ["test_id", "seq_time", "protocol"] + columns_numerical
        X, y = preprocess_2(training_data_names, data_dir)
    else:
        ap_0_sta_0 = [
            "ap_from_ap_1_mean_ant_rssi", "ap_from_ap_2_mean_ant_rssi", "sta_to_ap_0_mean_ant_rssi",
            "sta_to_ap_1_mean_ant_rssi", "sta_to_ap_2_mean_ant_rssi", "sta_from_ap_0_mean_ant_rssi",
            "sta_from_ap_1_mean_ant_rssi", "sta_from_ap_2_mean_ant_rssi", "sta_from_sta_1_rssi", "sta_from_sta_2_rssi"
        ]
        ap_1_sta_1 = [
            "ap_from_ap_0_mean_ant_rssi", "ap_from_ap_2_mean_ant_rssi", "sta_to_ap_0_mean_ant_rssi",
            "sta_to_ap_1_mean_ant_rssi", "sta_to_ap_2_mean_ant_rssi", "sta_from_ap_0_mean_ant_rssi",
            "sta_from_ap_1_mean_ant_rssi", "sta_from_ap_2_mean_ant_rssi", "sta_from_sta_0_rssi", "sta_from_sta_2_rssi"
        ]
        ap_2_sta_2 = [
            "ap_from_ap_0_mean_ant_rssi", "ap_from_ap_1_mean_ant_rssi", "sta_to_ap_0_mean_ant_rssi",
            "sta_to_ap_1_mean_ant_rssi", "sta_to_ap_2_mean_ant_rssi", "sta_from_ap_0_mean_ant_rssi",
            "sta_from_ap_1_mean_ant_rssi", "sta_from_ap_2_mean_ant_rssi", "sta_from_sta_0_rssi", "sta_from_sta_1_rssi"
        ]
        columns_basic = ["test_id", "seq_time", "protocol", "eirp", "nav"] + columns_numerical
        X, y = preprocess_3(training_data_names, data_dir)

    # loop_test(X, y, 0, 1, ap_name, output)
    model, y_pred, mse, scaler, X_train = model_predict(X, y)
    print(f"Mean Squared Error: {mse}")

    if ap_name == '2ap':
        ap_0_sta_0 = [
            "ap_from_ap_1_mean_ant_rssi", "sta_to_ap_0_mean_ant_rssi", "sta_to_ap_1_mean_ant_rssi",
            "sta_from_ap_0_mean_ant_rssi", "sta_from_ap_1_mean_ant_rssi"
        ]
        ap_1_sta_1 = [
            "ap_from_ap_0_mean_ant_rssi", "sta_to_ap_1_mean_ant_rssi", "sta_to_ap_0_mean_ant_rssi",
            "sta_from_ap_1_mean_ant_rssi", "sta_from_ap_0_mean_ant_rssi"
        ]
        predict_test_2(data_dir, ap_name, output, scaler, X_train)
    else:
        ap_0_sta_0 = [
            "ap_from_ap_1_mean_ant_rssi", "ap_from_ap_2_mean_ant_rssi", "sta_to_ap_0_mean_ant_rssi",
            "sta_to_ap_1_mean_ant_rssi", "sta_to_ap_2_mean_ant_rssi", "sta_from_ap_0_mean_ant_rssi",
            "sta_from_ap_1_mean_ant_rssi", "sta_from_ap_2_mean_ant_rssi"
        ]
        ap_1_sta_1 = [
            "ap_from_ap_0_mean_ant_rssi", "ap_from_ap_2_mean_ant_rssi", "sta_to_ap_0_mean_ant_rssi",
            "sta_to_ap_1_mean_ant_rssi", "sta_to_ap_2_mean_ant_rssi", "sta_from_ap_0_mean_ant_rssi",
            "sta_from_ap_1_mean_ant_rssi", "sta_from_ap_2_mean_ant_rssi"
        ]
        ap_2_sta_2 = [
            "ap_from_ap_0_mean_ant_rssi", "ap_from_ap_1_mean_ant_rssi", "sta_to_ap_0_mean_ant_rssi",
            "sta_to_ap_1_mean_ant_rssi", "sta_to_ap_2_mean_ant_rssi", "sta_from_ap_0_mean_ant_rssi",
            "sta_from_ap_1_mean_ant_rssi", "sta_from_ap_2_mean_ant_rssi"
        ]

    # 2ap 1929
    # 3ap 6103

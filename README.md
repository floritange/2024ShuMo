# 2024中国研究生数学建模B题
# 2024 Chinese Graduate Mathematical Modeling (Question B)


# 环境

```bash
pip install -e .
```

# 项目结构
其中

最终报告./docs/B24105580031.pdf

最终源码./docs/B24105580031.zip

解压data.7z

```
.
├── README.md
├── __init__.py
├── data
│   ├── B
│   │   ├── B.zip
│   │   ├── test_set_1_2ap.csv
│   │   ├── test_set_1_3ap.csv
│   │   ├── test_set_2_2ap.csv
│   │   ├── test_set_2_3ap.csv
│   │   ├── training_set_2ap_loc0_nav82.csv
│   │   ├── training_set_2ap_loc0_nav86.csv
│   │   ├── training_set_2ap_loc1_nav82.csv
│   │   ├── training_set_2ap_loc1_nav86.csv
│   │   ├── training_set_2ap_loc2_nav82.csv
│   │   ├── training_set_3ap_loc30_nav82.csv
│   │   ├── training_set_3ap_loc30_nav86.csv
│   │   ├── training_set_3ap_loc31_nav82.csv
│   │   ├── training_set_3ap_loc31_nav86.csv
│   │   ├── training_set_3ap_loc32_nav82.csv
│   │   ├── training_set_3ap_loc32_nav86.csv
│   │   ├── training_set_3ap_loc33_nav82.csv
│   │   └── training_set_3ap_loc33_nav88.csv
│   ├── data_error
│   │   └── training_set_2ap_loc2_nav82.csv
│   ├── processed
│   │   ├── training_data_all.csv
│   │   ├── training_data_all_2ap.csv
│   │   └── training_data_all_3ap.csv
│   ├── results
│   │   ├── B_add_column
│   │   │   ├── test_set_1_2ap.csv
│   │   │   ├── test_set_1_3ap.csv
│   │   │   ├── test_set_2_2ap.csv
│   │   │   ├── test_set_2_3ap.csv
│   │   │   ├── training_set_2ap_loc0_nav82.csv
│   │   │   ├── training_set_2ap_loc0_nav86.csv
│   │   │   ├── training_set_2ap_loc1_nav82.csv
│   │   │   ├── training_set_2ap_loc1_nav86.csv
│   │   │   ├── training_set_2ap_loc2_nav82.csv
│   │   │   ├── training_set_3ap_loc30_nav82.csv
│   │   │   ├── training_set_3ap_loc30_nav86.csv
│   │   │   ├── training_set_3ap_loc31_nav82.csv
│   │   │   ├── training_set_3ap_loc31_nav86.csv
│   │   │   ├── training_set_3ap_loc32_nav82.csv
│   │   │   ├── training_set_3ap_loc32_nav86.csv
│   │   │   ├── training_set_3ap_loc33_nav82.csv
│   │   │   └── training_set_3ap_loc33_nav88.csv
│   │   ├── eirp值临界点统计.xlsx
│   │   ├── question1
│   │   │   ├── test_set_1_2ap.csv
│   │   │   ├── test_set_1_3ap.csv
│   │   │   ├── test_set_2_2ap.csv
│   │   │   ├── test_set_2_3ap.csv
│   │   │   ├── training_set_2ap_loc0_nav82.csv
│   │   │   ├── training_set_2ap_loc0_nav86.csv
│   │   │   ├── training_set_2ap_loc1_nav82.csv
│   │   │   ├── training_set_2ap_loc1_nav86.csv
│   │   │   ├── training_set_2ap_loc2_nav82.csv
│   │   │   ├── training_set_3ap_loc30_nav82.csv
│   │   │   ├── training_set_3ap_loc30_nav86.csv
│   │   │   ├── training_set_3ap_loc31_nav82.csv
│   │   │   ├── training_set_3ap_loc31_nav86.csv
│   │   │   ├── training_set_3ap_loc32_nav82.csv
│   │   │   ├── training_set_3ap_loc32_nav86.csv
│   │   │   ├── training_set_3ap_loc33_nav82.csv
│   │   │   └── training_set_3ap_loc33_nav88.csv
│   │   ├── question1_add_column
│   │   │   ├── test_set_1_2ap.csv
│   │   │   ├── test_set_1_3ap.csv
│   │   │   ├── test_set_2_2ap.csv
│   │   │   ├── test_set_2_3ap.csv
│   │   │   ├── training_set_2ap_loc0_nav82.csv
│   │   │   ├── training_set_2ap_loc0_nav86.csv
│   │   │   ├── training_set_2ap_loc1_nav82.csv
│   │   │   ├── training_set_2ap_loc1_nav86.csv
│   │   │   ├── training_set_2ap_loc2_nav82.csv
│   │   │   ├── training_set_3ap_loc30_nav82.csv
│   │   │   ├── training_set_3ap_loc30_nav86.csv
│   │   │   ├── training_set_3ap_loc31_nav82.csv
│   │   │   ├── training_set_3ap_loc31_nav86.csv
│   │   │   ├── training_set_3ap_loc32_nav82.csv
│   │   │   ├── training_set_3ap_loc32_nav86.csv
│   │   │   ├── training_set_3ap_loc33_nav82.csv
│   │   │   └── training_set_3ap_loc33_nav88.csv
│   │   ├── question2
│   │   │   ├── test_set_1_2ap.csv
│   │   │   ├── test_set_1_3ap.csv
│   │   │   ├── test_set_2_2ap.csv
│   │   │   ├── test_set_2_3ap.csv
│   │   │   ├── training_set_2ap_loc0_nav82.csv
│   │   │   ├── training_set_2ap_loc0_nav86.csv
│   │   │   ├── training_set_2ap_loc1_nav82.csv
│   │   │   ├── training_set_2ap_loc1_nav86.csv
│   │   │   ├── training_set_2ap_loc2_nav82.csv
│   │   │   ├── training_set_3ap_loc30_nav82.csv
│   │   │   ├── training_set_3ap_loc30_nav86.csv
│   │   │   ├── training_set_3ap_loc31_nav82.csv
│   │   │   ├── training_set_3ap_loc31_nav86.csv
│   │   │   ├── training_set_3ap_loc32_nav82.csv
│   │   │   ├── training_set_3ap_loc32_nav86.csv
│   │   │   ├── training_set_3ap_loc33_nav82.csv
│   │   │   └── training_set_3ap_loc33_nav88.csv
│   │   └── question3
│   │       ├── test_set_1_2ap.csv
│   │       ├── test_set_2_2ap.csv
│   │       ├── training_set_2ap_loc0_nav82.csv
│   │       ├── training_set_2ap_loc0_nav86.csv
│   │       ├── training_set_2ap_loc1_nav82.csv
│   │       ├── training_set_2ap_loc1_nav86.csv
│   │       └── training_set_2ap_loc2_nav82.csv
│   └── test_eirp值临界点统计.xlsx
├── docs
│   ├── B24105580031.pdf
│   ├── B24105580031.zip
│   ├── WLAN组网中网络吞吐量建模.docx
│   ├── data_test.xlsx
│   ├── ~$AN组网中网络吞吐量建模.docx
│   ├── 草稿.txt
│   ├── 未命名.xlsx
│   ├── 预测结果
│   │   ├── 问题一
│   │   │   ├── test_set_1_2ap.csv
│   │   │   ├── test_set_1_2ap.png
│   │   │   ├── test_set_1_3ap.csv
│   │   │   └── test_set_1_3ap.png
│   │   ├── 问题三
│   │   │   ├── test_set_1_2ap.csv
│   │   │   ├── test_set_1_2ap.png
│   │   │   ├── test_set_1_3ap.csv
│   │   │   └── test_set_1_3ap.png
│   │   └── 问题二
│   │       ├── test_set_2_2ap.csv
│   │       ├── test_set_2_2ap.png
│   │       ├── test_set_2_3ap.csv
│   │       └── test_set_2_3ap.png
│   ├── 预测结果.zip
│   ├── “华为杯”第二十一届中国研究生数学建模竞赛论文V1.0.docx
│   ├── “华为杯”第二十一届中国研究生数学建模竞赛论文V1.0_副本.docx
│   ├── “华为杯”第二十一届中国研究生数学建模竞赛论文V2.2.docx
│   └── “华为杯”第二十一届中国研究生数学建模竞赛论文V3.0.docx
├── logs
│   └── project_running.log
├── setup.py
├── shumo
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-311.pyc
│   │   └── utils.cpython-311.pyc
│   ├── data_process
│   │   ├── 1question.py
│   │   ├── __init__.py
│   │   ├── add_column_test.ipynb
│   │   ├── add_column_train.ipynb
│   │   ├── data_preprocess.ipynb
│   │   ├── data_preprocess_select.ipynb
│   │   ├── data_process.zip
│   │   ├── question1.py
│   │   ├── question1_2ap.ipynb
│   │   ├── question1_2ap_addeirp.ipynb
│   │   ├── question1_3ap.ipynb
│   │   ├── question1_3ap_addeirp.ipynb
│   │   ├── question2_2ap.ipynb
│   │   ├── question2_3ap.ipynb
│   │   ├── question3_2ap.ipynb
│   │   └── test.ipynb
│   └── utils.py
└── shuomo.egg-info
    ├── PKG-INFO
    ├── SOURCES.txt
    ├── dependency_links.txt
    └── top_level.txt
```

常量特征：
test_dur,pkt_len,pd,ed,seq_time

ap和sta类别特征：
bss_id,ap_name,ap_mac,ap_id,sta_mac,sta_id,

变量特征：
loc_id,protocol,nav

测量统计指标：
nss,mcs,per,num_ampdu,ppdu_dur,other_air_time,seq_time,throughput

# 数据信息
## 测试基本信息:
test_id,test_dur,loc_id,bss_id,ap_name,ap_mac,ap_id,sta_mac,sta_id,protocol,pkt_len,pd,ed,nav,seq_time
data_raw_ap_0信息如下。提取下面这些列，test_id作为横轴，画所有曲线图，seq_time特别标出。画在一张图里
1.有列如果不为数值，先编码。
2.用min-max归一化

- 网络拓扑:
test_id,test_dur,loc_id,bss_id,ap_name,ap_mac,ap_id,sta_mac,sta_id,
- 业务流量：
protocol,pkt_len
- 门限信息：
pd,ed,nav
- 节点间RSSI信息：
eirp,ap_from_ap_0_sum_ant_rssi,ap_from_ap_0_max_ant_rssi,ap_from_ap_0_mean_ant_rssi,ap_from_ap_1_sum_ant_rssi,ap_from_ap_1_max_ant_rssi,ap_from_ap_1_mean_ant_rssi,sta_to_ap_0_sum_ant_rssi,sta_to_ap_0_max_ant_rssi,sta_to_ap_0_mean_ant_rssi,sta_to_ap_1_sum_ant_rssi,sta_to_ap_1_max_ant_rssi,sta_to_ap_1_mean_ant_rssi,sta_from_ap_0_sum_ant_rssi,sta_from_ap_0_max_ant_rssi,sta_from_ap_0_mean_ant_rssi,sta_from_ap_1_sum_ant_rssi,sta_from_ap_1_max_ant_rssi,sta_from_ap_1_mean_ant_rssi,sta_from_sta_0_rssi,sta_from_sta_1_rssi

## 数据帧统计信息
nss,mcs,per,num_ampdu,ppdu_dur,other_air_time,seq_time,throughput

## 目标变量
predict throughput,error

## 模型精度评估方法
CDF

## 所有数据
test_id,test_dur,loc_id,protocol,pkt_len,bss_id,ap_name,ap_mac,ap_id,pd,ed,nav,eirp,ap_from_ap_0_sum_ant_rssi,ap_from_ap_0_max_ant_rssi,ap_from_ap_0_mean_ant_rssi,ap_from_ap_1_sum_ant_rssi,ap_from_ap_1_max_ant_rssi,ap_from_ap_1_mean_ant_rssi,sta_mac,sta_id,sta_to_ap_0_sum_ant_rssi,sta_to_ap_0_max_ant_rssi,sta_to_ap_0_mean_ant_rssi,sta_to_ap_1_sum_ant_rssi,sta_to_ap_1_max_ant_rssi,sta_to_ap_1_mean_ant_rssi,sta_from_ap_0_sum_ant_rssi,sta_from_ap_0_max_ant_rssi,sta_from_ap_0_mean_ant_rssi,sta_from_ap_1_sum_ant_rssi,sta_from_ap_1_max_ant_rssi,sta_from_ap_1_mean_ant_rssi,sta_from_sta_0_rssi,sta_from_sta_1_rssi,nss,mcs,per,num_ampdu,ppdu_dur,other_air_time,seq_time,throughput

# question1
恒定指标:
test_dur, pkt_len, pd, ed
bss_id, ap_name, ap_mac, ap_id
sta_mac, sta_id


选取指标
loc_id：用来判断该位置信号干扰情况
nav: 判断AP是否发生传输
eirp: AP发射信号强度，达到阈值才能成功发送

观测：
protocol,

UDP和TCP协议对应sqe_time
文件顺序，2AP、3AP分开

### 进一步分析
1. eirp相关性分析：单个文件，两个AP曲线+eirp，拆成 tcp 和 udp
test_id, eirp, protocol, seq_time
2. loc_id相关性分析：2个AP纵向对比，3个AP分开对象对比，根据NAV分类，loc_id+seq_time
3. nav相关性分析

### 最终建模
1. eirp: 功率增大，干扰增强，序列时间下降
2. tcp 和 udp: 业务不同，tcp更稳定，udp更混乱，随着功率变化出现变化
3. loc_id: 不同的位置出现干扰的功率不同
4. nav: 随着功率变化拐点位置不同
目标：seq_time


5 2ap train
8 3ap train

2ap
sum_ant_rssi
max_ant_rssi
mean_ant_rssi
sum max mean_ant_rssi
ap_0: ap_from_ap_1
sta_0: sta_to_ap_0, sta_to_ap_1, sta_from_ap_0, sta_from_ap_1, sta_from_sta_1

ap_0_sta_0 = ["ap_from_ap_1_mean_ant_rssi", "sta_to_ap_0_mean_ant_rssi", "sta_to_ap_1_mean_ant_rssi", "sta_from_ap_0_mean_ant_rssi", "sta_from_ap_1_mean_ant_rssi", "sta_from_sta_1_rssi"]
ap_1_sta_1 = ["ap_from_ap_0_mean_ant_rssi", "sta_to_ap_1_mean_ant_rssi", "sta_to_ap_0_mean_ant_rssi", "sta_from_ap_1_mean_ant_rssi", "sta_from_ap_0_mean_ant_rssi", "sta_from_sta_0_rssi"]

ap_0_sta_0 = [
    "ap_from_ap_1_mean_ant_rssi", "sta_to_ap_0_mean_ant_rssi", "sta_to_ap_1_mean_ant_rssi", 
    "sta_from_ap_0_mean_ant_rssi", "sta_from_ap_1_mean_ant_rssi", "sta_from_sta_1_rssi",
    "ap_from_ap_1_max_ant_rssi", "sta_to_ap_0_max_ant_rssi", "sta_to_ap_1_max_ant_rssi", 
    "sta_from_ap_0_max_ant_rssi", "sta_from_ap_1_max_ant_rssi"
]
ap_1_sta_1 = [
    "ap_from_ap_0_mean_ant_rssi", "sta_to_ap_1_mean_ant_rssi", "sta_to_ap_0_mean_ant_rssi", 
    "sta_from_ap_1_mean_ant_rssi", "sta_from_ap_0_mean_ant_rssi", "sta_from_sta_0_rssi",
    "ap_from_ap_0_max_ant_rssi", "sta_to_ap_1_max_ant_rssi", "sta_to_ap_0_max_ant_rssi", 
    "sta_from_ap_1_max_ant_rssi", "sta_from_ap_0_max_ant_rssi"
]


ap_0_sta_0 = [
    "ap_from_ap_1_mean_ant_rssi", "sta_to_ap_0_mean_ant_rssi", "sta_to_ap_1_mean_ant_rssi", 
    "sta_from_ap_0_mean_ant_rssi", "sta_from_ap_1_mean_ant_rssi", "sta_from_sta_1_rssi",
    "ap_from_ap_1_sum_ant_rssi", "sta_to_ap_0_sum_ant_rssi", "sta_to_ap_1_sum_ant_rssi", 
    "sta_from_ap_0_sum_ant_rssi", "sta_from_ap_1_sum_ant_rssi",
    "ap_from_ap_1_max_ant_rssi", "sta_to_ap_0_max_ant_rssi", "sta_to_ap_1_max_ant_rssi", 
    "sta_from_ap_0_max_ant_rssi", "sta_from_ap_1_max_ant_rssi"
]
ap_1_sta_1 = [
    "ap_from_ap_0_mean_ant_rssi", "sta_to_ap_1_mean_ant_rssi", "sta_to_ap_0_mean_ant_rssi", 
    "sta_from_ap_1_mean_ant_rssi", "sta_from_ap_0_mean_ant_rssi", "sta_from_sta_0_rssi",
    "ap_from_ap_0_sum_ant_rssi", "sta_to_ap_1_sum_ant_rssi", "sta_to_ap_0_sum_ant_rssi", 
    "sta_from_ap_1_sum_ant_rssi", "sta_from_ap_0_sum_ant_rssi",
    "ap_from_ap_0_max_ant_rssi", "sta_to_ap_1_max_ant_rssi", "sta_to_ap_0_max_ant_rssi", 
    "sta_from_ap_1_max_ant_rssi", "sta_from_ap_0_max_ant_rssi"
]

============



================

3ap
ap_0: ap_from_ap_1, ap_from_ap_2
sta_0: sta_to_ap_0, sta_to_ap_1, sta_to_ap_2, sta_from_ap_0, sta_from_ap_1, sta_from_ap_2, sta_from_sta_1, sta_from_sta_2

ap_from_ap_1, ap_from_ap_2, sta_to_ap_0, sta_to_ap_1, sta_to_ap_2, sta_from_ap_0, sta_from_ap_1, sta_from_ap_2, sta_from_sta_1, sta_from_sta_2

ap_0_sta_0 = ["ap_from_ap_1_mean_ant_rssi", "ap_from_ap_2_mean_ant_rssi", "sta_to_ap_0_mean_ant_rssi", "sta_to_ap_1_mean_ant_rssi", "sta_to_ap_2_mean_ant_rssi", "sta_from_ap_0_mean_ant_rssi", "sta_from_ap_1_mean_ant_rssi", "sta_from_ap_2_mean_ant_rssi", "sta_from_sta_1_rssi", "sta_from_sta_2_rssi"]
ap_1_sta_1 = ["ap_from_ap_0_mean_ant_rssi", "ap_from_ap_2_mean_ant_rssi", "sta_to_ap_0_mean_ant_rssi", "sta_to_ap_1_mean_ant_rssi", "sta_to_ap_2_mean_ant_rssi", "sta_from_ap_0_mean_ant_rssi", "sta_from_ap_1_mean_ant_rssi", "sta_from_ap_2_mean_ant_rssi", "sta_from_sta_0_rssi", "sta_from_sta_2_rssi"]
ap_2_sta_2 = ["ap_from_ap_0_mean_ant_rssi", "ap_from_ap_1_mean_ant_rssi", "sta_to_ap_0_mean_ant_rssi", "sta_to_ap_1_mean_ant_rssi", "sta_to_ap_2_mean_ant_rssi", "sta_from_ap_0_mean_ant_rssi", "sta_from_ap_1_mean_ant_rssi", "sta_from_ap_2_mean_ant_rssi", "sta_from_sta_0_rssi", "sta_from_sta_1_rssi"]


ap_0_sta_0 = [
    "ap_from_ap_1_mean_ant_rssi", "ap_from_ap_2_mean_ant_rssi", 
    "sta_to_ap_0_mean_ant_rssi", "sta_to_ap_1_mean_ant_rssi", "sta_to_ap_2_mean_ant_rssi", 
    "sta_from_ap_0_mean_ant_rssi", "sta_from_ap_1_mean_ant_rssi", "sta_from_ap_2_mean_ant_rssi", 
    "sta_from_sta_1_rssi", "sta_from_sta_2_rssi",
    "ap_from_ap_1_sum_ant_rssi", "ap_from_ap_2_sum_ant_rssi", 
    "sta_to_ap_0_sum_ant_rssi", "sta_to_ap_1_sum_ant_rssi", "sta_to_ap_2_sum_ant_rssi", 
    "sta_from_ap_0_sum_ant_rssi", "sta_from_ap_1_sum_ant_rssi", "sta_from_ap_2_sum_ant_rssi", 
    "ap_from_ap_1_max_ant_rssi", "ap_from_ap_2_max_ant_rssi", 
    "sta_to_ap_0_max_ant_rssi", "sta_to_ap_1_max_ant_rssi", "sta_to_ap_2_max_ant_rssi", 
    "sta_from_ap_0_max_ant_rssi", "sta_from_ap_1_max_ant_rssi", "sta_from_ap_2_max_ant_rssi"
]
ap_1_sta_1 = [
    "ap_from_ap_0_mean_ant_rssi", "ap_from_ap_2_mean_ant_rssi", 
    "sta_to_ap_0_mean_ant_rssi", "sta_to_ap_1_mean_ant_rssi", "sta_to_ap_2_mean_ant_rssi", 
    "sta_from_ap_0_mean_ant_rssi", "sta_from_ap_1_mean_ant_rssi", "sta_from_ap_2_mean_ant_rssi", 
    "sta_from_sta_0_rssi", "sta_from_sta_2_rssi",
    "ap_from_ap_0_sum_ant_rssi", "ap_from_ap_2_sum_ant_rssi", 
    "sta_to_ap_0_sum_ant_rssi", "sta_to_ap_1_sum_ant_rssi", "sta_to_ap_2_sum_ant_rssi", 
    "sta_from_ap_0_sum_ant_rssi", "sta_from_ap_1_sum_ant_rssi", "sta_from_ap_2_sum_ant_rssi", 
    "ap_from_ap_0_max_ant_rssi", "ap_from_ap_2_max_ant_rssi", 
    "sta_to_ap_0_max_ant_rssi", "sta_to_ap_1_max_ant_rssi", "sta_to_ap_2_max_ant_rssi", 
    "sta_from_ap_0_max_ant_rssi", "sta_from_ap_1_max_ant_rssi", "sta_from_ap_2_max_ant_rssi"
]
ap_2_sta_2 = [
    "ap_from_ap_0_mean_ant_rssi", "ap_from_ap_1_mean_ant_rssi", 
    "sta_to_ap_0_mean_ant_rssi", "sta_to_ap_1_mean_ant_rssi", "sta_to_ap_2_mean_ant_rssi", 
    "sta_from_ap_0_mean_ant_rssi", "sta_from_ap_1_mean_ant_rssi", "sta_from_ap_2_mean_ant_rssi", 
    "sta_from_sta_0_rssi", "sta_from_sta_1_rssi",

    "ap_from_ap_0_sum_ant_rssi", "ap_from_ap_1_sum_ant_rssi", 
    "sta_to_ap_0_sum_ant_rssi", "sta_to_ap_1_sum_ant_rssi", "sta_to_ap_2_sum_ant_rssi", 
    "sta_from_ap_0_sum_ant_rssi", "sta_from_ap_1_sum_ant_rssi", "sta_from_ap_2_sum_ant_rssi", 

    "ap_from_ap_0_max_ant_rssi", "ap_from_ap_1_max_ant_rssi", 
    "sta_to_ap_0_max_ant_rssi", "sta_to_ap_1_max_ant_rssi", "sta_to_ap_2_max_ant_rssi", 
    "sta_from_ap_0_max_ant_rssi", "sta_from_ap_1_max_ant_rssi", "sta_from_ap_2_max_ant_rssi"
]



training_set_2ap_loc2_nav82.csv
最后一个点是异常点
39,60,loc2,udp,1500,0,model-3,8c68-3a11-e370,ap_0,-82,-62,-82,28,,,,"[-66, -68, -68, -68, -68, -68, -68, -67, -69, -71, -68, -68, -70, -67, -68, -67, -68, -67, -68, -68, -67, -77, -67, -68, -86, -68, -68, -67, -67, -68, -73, -67, -71, -68, -68, -67, -68, -71, -68, -86, -69, -68, -67, -69, -70, -68, -67, -68, -68, -70, -68, -68, -68, -68, -67, -66, -68, -68, -67, -68, -68, -86, -68, -67, -85, -68, -68, -68, -67, -68, -67, -68, -67, -67, -66, -67, -67, -69, -68, -67, -67, -68, -68, -68, -69, -69, -67, -68, -68]","[-73, -75, -73, -74, -75, -73, -75, -73, -76, -73, -72, -75, -73, -73, -76, -71, -75, -74, -75, -76, -72, -77, -72, -76, -92, -76, -76, -74, -72, -74, -75, -73, -75, -76, -73, -73, -75, -76, -75, -92, -75, -75, -75, -74, -73, -75, -72, -76, -75, -75, -74, -75, -74, -75, -74, -70, -74, -73, -74, -75, -75, -94, -76, -73, -91, -74, -74, -75, -73, -75, -73, -73, -74, -71, -71, -71, -73, -76, -73, -73, -74, -75, -75, -75, -75, -74, -74, -75, -73, -94, -94, -94, -94]","[-76, -77, -77, -77, -77, -77, -77, -76, -78, -80, -77, -77, -79, -76, -77, -76, -77, -76, -77, -77, -76, -86, -76, -77, -95, -78, -77, -76, -76, -77, -82, -76, -80, -77, -77, -76, -77, -80, -77, -95, -78, -77, -76, -76, -79, -77, -76, -77, -77, -79, -77, -77, -77, -77, -76, -75, -77, -77, -76, -77, -77, -95, -77, -76, -94, -77, -77, -77, -76, -77, -76, -77, -76, -76, -75, -76, -76, -78, -77, -76, -76, -77, -77, -77, -78, -78, -76, -77, -77]",d4e0,sta_0,"[-65, -62, -61, -62, -62, -62, -66, -62, -62, -62, -63, -63, -62, -62, -63, -62, -62, -62, -63, -62, -62, -63, -62, -62, -62, -62, -62, -68, -62, -61, -62, -62, -63, -63, -68, -62, -62, -62, -63, -68, -63, -62, -61, -76, -63, -61, -62, -63, -62, -62, -63, -63, -62, -63, -62, -71, -62, -63, -63, -62, -62, -63, -66, -63, -63, -63, -63, -54, -62, -67, -63, -63, -62, -60, -63, -62, -62, -62, -62, -62, -62]","[-69, -94, -67, -65, -65, -67, -65, -69, -67, -69, -66, -65, -67, -68, -67, -66, -66, -67, -67, -68, -67, -67, -68, -66, -67, -66, -67, -68, -72, -66, -64, -66, -65, -65, -68, -73, -66, -66, -67, -66, -69, -69, -68, -66, -76, -68, -65, -67, -67, -66, -65, -67, -68, -66, -66, -66, -75, -68, -66, -69, -65, -65, -66, -71, -68, -67, -70, -67, -58, -66, -72, -67, -68, -66, -66, -67, -66, -66, -66, -67, -66, -67, -93, -94, -94]","[-73, -71, -70, -71, -71, -71, -75, -71, -71, -71, -70, -72, -71, -71, -72, -71, -71, -71, -72, -71, -71, -72, -71, -71, -71, -71, -71, -75, -71, -70, -71, -71, -70, -72, -74, -71, -71, -71, -72, -77, -72, -71, -70, -79, -72, -70, -71, -72, -71, -70, -72, -72, -71, -69, -71, -80, -71, -72, -72, -71, -71, -70, -73, -72, -72, -72, -72, -63, -71, -73, -72, -72, -71, -69, -72, -71, -71, -68, -71, -71, -71]","[-85, -86, -87, -87, -87, -77, -87, -87, -72, -87, -72, -87, -87, -71, -87, -87, -87, -86, -87, -87, -87, -87, -87, -87, -87, -71, -87, -87, -72, -72, -86, -87, -87, -87, -87, -87, -87, -87, -87, -87, -72, -72, -87, -87, -87, -87, -87, -87, -87, -87, -87, -87, -87, -87, -86, -72, -87, -87, -87, -87, -87, -86, -87, -87]","[-93, -93, -94, -94, -93, -94, -77, -94, -72, -93, -75, -93, -73, -94, -94, -93, -94, -94, -93, -94, -94, -94, -74, -93, -93, -72, -74, -92, -94, -93, -94, -93, -94, -93, -94, -94, -94, -93, -74, -74, -94, -93, -94, -94, -93, -93, -94, -94, -93, -92, -93, -94, -92, -72, -94, -94, -94, -94, -93, -94, -94]","[-94, -95, -96, -96, -96, -85, -96, -96, -80, -96, -81, -96, -96, -80, -96, -96, -96, -95, -96, -96, -96, -96, -96, -96, -80, -96, -96, -80, -80, -96, -96, -96, -96, -96, -96, -96, -96, -95, -96, -81, -80, -96, -96, -96, -96, -96, -96, -96, -96, -96, -96, -96, -96, -96, -80, -96, -96, -96, -96, -96, -95, -96, -96]","[-49, -49, -49, -49, -49, -48, -48, -49, -48, -48, -48, -48, -48, -48, -48, -48, -48, -48, -48, -48, -48, -48, -48, -48, -48, -48, -48, -48, -49, -49, -48, -48, -48, -48, -48, -48, -48, -48, -48, -48]","[-51, -51, -51, -51, -51, -50, -50, -50, -50, -50, -50, -50, -50, -50, -50, -50, -50, -50, -50, -50, -50, -50, -50, -50, -50, -50, -50, -50, -50, -50, -50, -50, -50, -50, -50, -50, -50, -50, -50, -50]","[-53, -53, -53, -53, -53, -52, -52, -52, -52, -52, -52, -52, -52, -52, -52, -52, -52, -52, -52, -52, -52, -52, -52, -52, -52, -52, -52, -52, -52, -52, -52, -52, -52, -52, -52, -52, -52, -52, -52, -52]","[-75, -75, -75, -75, -75, -74, -73, -73, -73, -73, -73, -73, -73, -73, -73, -73, -73, -73, -73, -73, -73, -73, -73, -73, -73, -73, -75, -74, -73, -74, -74, -74, -74, -74, -74, -74, -74, -74]","[-77, -77, -77, -77, -77, -76, -75, -75, -75, -75, -75, -75, -75, -75, -75, -75, -75, -75, -75, -75, -75, -75, -75, -75, -75, -75, -77, -75, -75, -76, -76, -76, -76, -76, -76, -76, -76, -76]","[-78, -78, -78, -78, -78, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -78, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77]",,-79,1,7,1,9,0.001577894,0,0.4,1.47


test_id,test_dur,loc_id,protocol,pkt_len,bss_id,ap_name,ap_mac,ap_id,pd,ed,nav,eirp,ap_from_ap_0_sum_ant_rssi,ap_from_ap_0_max_ant_rssi,ap_from_ap_0_mean_ant_rssi,ap_from_ap_1_sum_ant_rssi,ap_from_ap_1_max_ant_rssi,ap_from_ap_1_mean_ant_rssi,sta_mac,sta_id,sta_to_ap_0_sum_ant_rssi,sta_to_ap_0_max_ant_rssi,sta_to_ap_0_mean_ant_rssi,sta_to_ap_1_sum_ant_rssi,sta_to_ap_1_max_ant_rssi,sta_to_ap_1_mean_ant_rssi,sta_from_ap_0_sum_ant_rssi,sta_from_ap_0_max_ant_rssi,sta_from_ap_0_mean_ant_rssi,sta_from_ap_1_sum_ant_rssi,sta_from_ap_1_max_ant_rssi,sta_from_ap_1_mean_ant_rssi,sta_from_sta_0_rssi,sta_from_sta_1_rssi,nss,mcs,per,num_ampdu,ppdu_dur,other_air_time,seq_time,throughput,predict seq_time,add_change,predict nss,predict mcs


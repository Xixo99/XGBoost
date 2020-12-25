import numpy as np

# 加载训练集数据并修正标签为-1和1
x_train = np.load('./data/train_data.npy')
y_train = 2 * np.load('./data/train_target.npy') - 1
x_test = np.load('./data/test_data.npy')
y_test = 2 * np.load('./data/test_target.npy') - 1
# print(X_train, Y_train)

num_feature = x_train.shape[0]
num_data = x_train.shape[1]

# 排序
ordered_data = np.zeros([num_feature, 3, num_data])
for i in range(num_feature):
    # x
    ordered_data[i][0] = np.sort(x_train[i])
    order_index = np.argsort(x_train[i])
    # y
    ordered_data[i][1] = y_train[order_index]
    ordered_data[i][2] = order_index

# 初始的预测（全为0）
Pre = np.zeros(num_data)

# 选用方差做误差函数
def loss(pre, label):
    return 2 * (pre - label)


# 信息增益
def infgain(g, split, array_num):
    Gl = np.sum(g[0:split])
    Gr = np.sum(g[split:array_num])
    Gn = Gl + Gr
    Hl = 2 * split
    Hr = 2 * (array_num - split)
    Hn = 2 * array_num
    entropy_l = Gl * Gl / (Hl + Lambda)
    entropy_r = Gr * Gr / (Hr + Lambda)
    entropy_n = Gn * Gn / (Hn + Lambda)
    gain = entropy_l + entropy_r - entropy_n
    return gain


# 计算权重
def W(g, split, array_num):
    Gl = np.sum(g[0:split])
    Gr = np.sum(g[split:array_num])
    Hl = 2 * split
    Hr = 2 * (array_num - split)
    Wl = -Gl / (Lambda + Hl)
    Wr = -Gr / (Lambda + Hr)
    return Wl, Wr


# 预测
def predict(split, x, wl, wr, order):
    array_num = x.shape[0]
    # 乘上权重
    yl = x[0:split] * wl
    yr = x[split:array_num] * wr
    # 拼起来
    y = np.hstack([yl, yr])
    # 按位置加上
    order = order.astype(np.uint64)
    for i in range(array_num):
        order_item = order[i]
        Pre[order_item] += y[i]


# 生成叶子节点
def gen_leaf(data, pre):
    array_num = data.shape[2]
    best = np.zeros([2, num_feature])
    g = np.zeros([num_feature, array_num])
    gain = np.zeros([num_feature, array_num - 1])
    for featre_index in range(num_feature):
        for data_index in range(array_num):
            g[featre_index][data_index] = loss(pre[data_index], data[featre_index][1][data_index])
        for split in range(1, array_num):
            gain[featre_index][split - 1] = infgain(g, split, array_num)
        max_gain = np.max(gain[featre_index])

        best[0, featre_index] = max_gain
        best[1, featre_index] = np.where(gain[featre_index] == max_gain)[0][0] + 1

    max_feature = np.max(best[0])
    best_feature_index = np.where(best[0] == max_feature)[0][0]
    best_split_index = best[1][best_feature_index].astype(np.uint64)
    best_g = g[best_feature_index]

    wl, wr = W(best_g, best_split_index, array_num)

    predict(best_split_index, data[best_feature_index][0], wl, wr, data[best_feature_index][2])
    leaf = {'feature': best_feature_index,
            'split': best_split_index,
            'wl': wl,
            'wr': wr,
            'gain': max_gain
            }
    return pre, leaf


def train(data, pre):
    data_num = int(data.shape[2])
    t_pre, leaf = gen_leaf(data, pre)
    gain = leaf['gain']
    split = leaf['split']
    data_l = data[:, :, 0:split]
    data_r = data[:, :, split:data_num]
    pre_l = t_pre[0:split]
    pre_r = t_pre[split:data_num]
    resultl = {'data': data_l, 'gain': gain, 'pre': pre_l, 'split': split}
    resultr = {'data': data_r, 'gain': gain, 'pre': pre_r, 'split': data_num - split}
    return resultl, resultr


# 训练过程（递归调用）
def main_process(data, pre):
    if pre.shape[0] < 1:
        return
    l, r = train(data, pre)
    if l['gain'] > gate:
        main_process(l['data'], l['pre'])
    if r['gain'] > gate:
        main_process(r['data'], r['pre'])
    return


# 超参数
Lambda = 0.01
gate = 0

# 执行
if __name__ == '__main__':
    # 训练
    main_process(ordered_data, Pre)
    # 训练结果
    num = 0
    for i in range(num_data):
        if Pre[i] * y_train[i] > 0:
            num += 1
    print('acc:', num / num_data)

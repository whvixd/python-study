import csv, time, sys, pickle, h2o
from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials

'''
h2o 直接用 pip安装

demo：
    @link https://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/intro.html#installing-h2o-3
    
h2o暂时对CNN和RNN不支持，后续可能会将tensorflow集成进去
'''

#  this will attempt to discover an H2O at localhost:54321
h2o.init()
df = h2o.import_file(path='/Users/didi/Downloads/prostate.csv')


def init():
    # 计算列均值
    # df_mean = df.mean()
    # print(df_mean)

    vol = df['VOL']

    # 0 VOL 列 为缺失值
    vol[vol == 0] = None


def glm_demo():
    from h2o.estimators.glm import H2OGeneralizedLinearEstimator

    # 让第二列转为枚举类型
    df[1] = df[1].asfactor()

    # 定义模型 binomial：二项式
    m = H2OGeneralizedLinearEstimator(family="binomial")

    # <class 'h2o.estimators.glm.H2OGeneralizedLinearEstimator'>
    m.__class__

    # 训练模型
    m.train(x=df.names[2:], y='CAPSULE', training_frame=df)

    print("m_print:", m)


def gbm_demo():
    from h2o.estimators.gbm import H2OGradientBoostingEstimator

    df[1] = df[1].asfactor()

    m = H2OGradientBoostingEstimator(ntrees=10, max_depth=5)

    m.train(x=df.names[2:], y='CAPSULE', training_frame=df)

    print('m.type_print:', m.type)


def dl_demo():
    from h2o.estimators.deeplearning import H2ODeepLearningEstimator
    df[1] = df[1].asfactor()

    # 随机统一数字，每行一个
    random = df[0].runif()

    # 60%的训练集
    train = df[random < 0.6]

    # 30%的验证集
    valid = df[0.6 <= random < 0.9]

    # 10%的测试集
    test = df[random >= 0.9]

    m = H2ODeepLearningEstimator()

    print('m.train_print:',m.train(x=train.names[2:],y=train.names[1],training_frame=train,validation_frame=valid))
    print('m.train_print_end')

    print('m_print:',m)

    # 预测
    print('m.predict_print:\n',m.predict(test))
    print('m.predict_print_end')


    # 在训练数据上显示性能
    m.model_performance()

    # 在验证数据上显示性能
    m.model_performance(valid=True)

    # 评分并计算测试数据的新指标!
    print('m.model_performance(test_data=test)_print:',m.model_performance(test_data=test))
    print('m.model_performance(test_data=test)_print_end')

    # 训练数据的均方差
    m.mse()

    # 验证集上的均方差
    print('m.mse_print:',m.mse(valid=True))

    m.r2()
    print('m.r2_print:',m.r2(valid=True))

    print('m.confusion_matrix_print:',m.confusion_matrix())

    # 混淆矩阵的最大精度
    m.confusion_matrix(metrics="accuracy")

    # check out the help for more!
    m.confusion_matrix("min_per_class_accuracy")


if __name__ == '__main__':
    init()

    # glm_demo()
    # gbm_demo()
    dl_demo()

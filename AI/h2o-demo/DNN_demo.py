import csv, time, sys, pickle, h2o
from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials

'''
h2o 直接用 pip安装

demo：
    @link https://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/intro.html#installing-h2o-3
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


if __name__ == '__main__':
    init()

    glm_demo()

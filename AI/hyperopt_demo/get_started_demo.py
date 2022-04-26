import hyperopt,pickle
from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials
import matplotlib.pyplot as plt
import numpy as np

'''
hyperopt 定义：
    是进行超参数优化的一个类库。有了它我们就可以拜托手动调参的烦恼，并且往往能够在相对较短的时间内获取原优于手动调参的最终结果。
    
一般而言，使用hyperopt的方式的过程可以总结为：

    1、用于最小化的目标函数
    2、搜索空间
    3、存储搜索过程中所有点组合以及效果的方法
    4、要使用的搜索算法
'''


def demo_01():
    def objective(args):
        case, val = args
        return val if case == 'case_1' else val ** 2

    # 返回传入的列表或者数组其中的一个选项。
    space = hp.choice('a',
                      [('case_1', 1 + hp.lognormal('c1', 0, 1)),
                       ('case_2', hp.uniform('c2', -10, 10))])


    best = fmin(objective, space, algo=tpe.suggest, max_evals=100)

    print(best)

    print(hyperopt.space_eval(space, best))


def demo_02():
    trials = Trials()
    # 返回一个字典
    best = fmin(
        # 目标函数
        fn=lambda x: x,
        # 搜索空间，它有三个参数：名称x，范围的下限和上限0和1。
        space=hp.uniform('x', -2.5, 2.5),
        # algo参数指定搜索算法，本例中tpe表示 tree of Parzen estimators
        algo=tpe.suggest,
        # 执行的最大评估次数
        max_evals=100,
        trials=trials)
    print(best)

    print(trials.losses())

    # 以二进制的方式持久化存储
    with open('result/demo2_best.pkl', 'wb+') as result:
        pickle.dump(best, result)
    with open('result/demo2_trials.pkl', 'wb+') as result:
        pickle.dump(trials, result)


def demo_03():

    def fun_change(x):
        y = (np.sin(x - 2)) ** 2 * np.e ** (-x ** 2)
        return -y

    best = fmin(
        fn=fun_change,
        space=hp.uniform('x', -2.5, 2.5),
        algo=tpe.suggest,
        max_evals=100)
    print(best)

    x = np.linspace(-2.5, 2.5, 256, endpoint=True)  # 绘制X轴（-2.5,2）的图像

    f = (np.sin(x - 2)) ** 2 * np.e ** (-x ** 2)  # y值

    plt.plot(x, f, "g-", lw=2.5, label="f(x)")
    plt.scatter(best['x'], -fun_change(best['x']), 50, color='blue')
    plt.title('f(x) = sin^2(x-2)e^{-x^2}')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    demo_02()

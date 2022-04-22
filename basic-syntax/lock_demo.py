import threading
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


class Account:
    # 定义构造器
    def __init__(self, account_no, balance):
        # 封装账户编号、账户余额的两个成员变量
        self.account_no = account_no
        self.balance = balance
        self.lock = threading.RLock()


# 定义一个函数来模拟取钱操作
def draw(account: Account, draw_amount: int):
    account.lock.acquire()
    try:
        # 账户余额大于取钱数目
        if account.balance >= draw_amount:
            # 吐出钞票
            print(threading.current_thread().name \
                  + "取钱成功！吐出钞票:" + str(draw_amount))
            time.sleep(0.001)
            # 修改余额
            account.balance -= draw_amount
            return "\t余额为: " + str(account.balance)
        else:
            return threading.current_thread().name \
                   + "取钱失败！余额不足！"

    finally:
        account.lock.release()


# 创建一个账户
acct = Account("1234567", 1000)
# 创建线程池
threadPool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="test_")

# 进程池
processPool = ProcessPoolExecutor(max_workers=2)
for i in range(5):
    future = threadPool.submit(draw, acct, 800)
    print(future.result())
    # threading.Thread(name='user-%i' % i, target=draw, args=(acct, 800)).start()  # 定义一个函数来模拟取钱操作
threadPool.shutdown()

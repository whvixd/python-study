import _thread
import threading
import time


def func_(thread_name, action):
    for i in range(4):
        print(thread_name + " is doing " + action)


class ThreadDemo(threading.Thread):

    def __init__(self, thread_id: int, thread_name: str = None):
        threading.Thread.__init__(self)
        self.thread_id = thread_id
        self.thread_name = thread_name

    def run(self) -> None:
        for i in range(5):
            print("thrad_id:%i,thrad_name:%s" % (self.thread_id, self.thread_name))
            time.sleep(1)


if __name__ == '__main__':
    # 不推荐使用
    _thread.start_new_thread(func_, ("t1", "ac1"))
    _thread.start_new_thread(func_, ("t2", "ac2"))

    # 推荐使用
    # 重载的实现
    demo1 = ThreadDemo(1, 'demo1')
    demo2 = threading.Thread(target=func_, args=("demo2", "test"), daemon=True)
    demo1.start()
    demo2.start()

    # 主线程堵塞，等待demo1/2 结束
    demo1.join()
    demo2.join()
    print("demo1.daemon:", demo1.daemon)
    print("Main Thread end!")

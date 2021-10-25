def f_test():
    for i in range(10):
        if i % 2 == 0:
            yield i  # 暂时返回，不会结束程序
        print("next")


# r=f_test()
# print(list(r))
for i in f_test():
    print(i)

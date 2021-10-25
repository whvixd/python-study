counter: int = 0


def add_():
    # counter为全局参数，全局变量的值不能被随意调用与更改，如需调用，必须通过特殊方式声明后才能调用。
    global counter
    counter = counter + 1


add_()
print(counter)
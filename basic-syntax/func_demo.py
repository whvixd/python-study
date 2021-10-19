def func_(is_print: bool = False, name: str = "张三") -> str:
    print(is_print, name)
    return name


if __name__ == '__main__':
    name = func_(is_print=True)
    if name is None:
        print("none")
    print(name)

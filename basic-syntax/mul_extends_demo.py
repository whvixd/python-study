class Person():
    def __init__(self, name: str = None, age: int = 0):
        self.name = name
        self.age = age

    def action(self, action: str = None):
        print("%s with %i age is doing %s" % (self.name, self.age, action))


class Action():
    def __init__(self, action):
        self.action = action

    def do(self):
        print("do:%s" % self.action)




class Tom(Person, Action):
    # 多继承 MRO 算法
    def __init__(self):
        super().__init__("Tom", 20)
        super(Person,self).__init__("run")
        self.nickname = "smart"

    def tom_action(self):
        super(Tom, self).action("running")
        super().do()

    def super(cls, inst):
        mro = inst.__class__.mro()
        return mro[mro.index(cls) + 1]



if __name__ == '__main__':
    tom = Tom()
    tom.tom_action()

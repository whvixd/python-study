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

    def funca(self,*args, **kwargs):
        print(args)
        print(kwargs)

from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict, List

def _forward_unimplemented(self, *input: Any) -> None:
    raise NotImplementedError



class Model:

    forward:Callable[...,Any]=_forward_unimplemented

    def _call_impl(self,*args, **kwargs):
        print(args)
        print(kwargs)
        self.forward(args,kwargs)


    __call__: Callable[..., Any] = _call_impl


class Model_C(Model):
    def forward(self,*args, **kwargs):
        print("Model_C")
        print(args,kwargs)


if __name__ == '__main__':
    # tom = Tom()
    # tom.tom_action()
    # tom.funca(1,2,3,4,a=1,b=2)

    c=Model_C()
    c(1,2,3,input=[1,2,3])

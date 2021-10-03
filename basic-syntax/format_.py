# 保留浮点数小数点后两位
print('{:.2f}'.format(3.1415926))
# 将小数部分四舍五入化为整数
print('{:.0f}'.format(2.49828))
# 宽度为五左补x，x指任意字符也可以是空格
print('{:x>5.1f}'.format(2.4))
# 宽度为五右补x，x指任意字符也可以是空格
print('{:x<5.1f}'.format(2.4))
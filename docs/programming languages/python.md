## python实用手册

#### python类型




#### 打印输出
```py
print(value, ..., sep=' ', end='\n', file=sys.stdout, flush=False)
```
print()内建函数用于打印输出，默认打印到标准输出 sys.stdout。
```py
print("Hello %s: %d" % ("World", 100))
print("end")

>>>
Hello World: 100
end
```
Python 使用单引号或者双引号来表示字符，那么当打印含有单双引号的行时如何处理呢？
```py
print("It's a dog!")
print('It is a "Gentleman" dog!')
print('''It's a "Gentleman" dog!''')

>>>
It's a dog!
It is a "Gentleman" dog!
It's a "Gentleman" dog!
```
通过print()函数可以直接实现左对齐输出。print() 函数不能动态指定对齐的字符数， 也不能指定其他填充字符，只能使用默认的空格进行填充。
```py
man = [["Name", "John"], ["Age", "25"], ["Address", "BeiJing China"]]
for i in man:
    print("%-10s: %s" % (i[0], i[1]))

>>>
Name      : John
Age       : 25
Address   : BeiJing China
```
Python中字符串处理函数 ljust(), rjust() 和 center() 提供了更强大的对齐输出功能。
```py
print("123".ljust(5) == "123  ")
print("123".rjust(5) == "  123")
print("123".center(5) == " 123 ")

print("123".ljust(5, '~'))
print("123".rjust(5, '~'))
print("123".center(5, '~'))

>>>
True
True
True
123~~
~~123
~123~
```
##### 格式化输出
整数格式化符号可以指定不同进制：
<ul>
    <li>%o —— oct 八进制</li>
    <li>%d —— dec 十进制</li>
    <li>%x —— hex 十六进制</li>
    <li>%X —— hex 十六进制大写</li>
</ul>

```py
print('%o %d %x %X' % (10, 10, 10, 10))

>>>
12 10 a A
```
浮点数可以指定保留的小数位数或使用科学计数法：
<ul>
    <li>%f —— 保留小数点后面 6 位有效数字，%.2f，保留 2 位小数位。</li>
    <li>%e —— 保留小数点后面 6 位有效数字，指数形式输出，%.2e，保留 2 位小数位，使用科学计数法。</li>
</ul>

```py
print('%f' % 1.23)  # 默认保留6位小数
>>>
1.230000
print('%0.2f' % 1.23456) # 保留 2 位小数
>>>
1.23
print('%e' % 1.23)  # 默认6位小数，用科学计数法
>>>
1.230000e+00
print('%0.2e' % 1.23)  # 保留 2 位小数，用科学计数法
>>>
1.23e+00
```

字符串格式化
<ul>
    <li>%s —— 格式化字符串</li>
    <li>%10s —— 右对齐，空格占位符 10 位</li>
    <li>%-10s —— 左对齐，空格占位符 10 位</li>
    <li>%.2s —— 截取 2 个字符串</li>
    <li>%10.2s —— 10 位占位符，截取两个字符</li>
</ul>

```py
print('%s' % 'hello world')  # 字符串输出
>>>
hello world
print('%20s' % 'hello world')  # 右对齐，取 20 个字符，不够则空格补位
>>>
         hello world
print('%-20s' % 'hello world')  # 左对齐，取 20 个字符，不够则空格补位
>>>
hello world
print('%.2s' % 'hello world')  # 取 2 个字符，默认左对齐
>>>
he
print('%10.2s' % 'hello world')  # 右对齐，取 2 个字符
>>>
        he
print('%-10.2s' % 'hello world')  # 左对齐，取 2 个字符
>>>
he
```

###### format 格式化
format() 是字符串对象的内置函数，它提供了比百分号格式化更强大的功能，例如调整参数顺序，支持字典关键字等。它该函数把字符串当成一个模板，通过传入的参数进行格式化，并且使用大括号 ‘{}’ 作为特殊字符代替 ‘%’。

1. 位置匹配
    位置匹配有以下几种方式：
    * 不带编号，即“{}”，此时按顺序匹配
    * 带数字编号，可调换顺序，即 “{1}”、“{2}”，按编号匹配
    * 带关键字，即“{name}”、“{name1}”，按字典键匹配
    * 通过对象属性匹配，例如 obj.x
    * 通过下标索引匹配，例如 a[0]，a[1]
```py
>>> print('{} {}'.format('hello','world'))  # 默认从左到右匹配
hello world
>>> print('{0} {1}'.format('hello','world'))  # 按数字编号匹配
hello world
>>> print('{0} {1} {0}'.format('hello','world'))  # 打乱顺序
hello world hello
>>> print('{1} {1} {0}'.format('hello','world'))
world world hello
>>> print('{wd} {ho}'.format(ho='hello',wd='world'))  # 关键字匹配
world hello
```
对于元组，列表，字典等支持索引的对象，支持使用索引匹配位置：
```py
>>> point = (0, 1)
>>> 'X: {0[0]};  Y: {0[1]}'.format(point)
'X: 0;  Y: 1'
>>> a = {'a': 'val_a', 'b': 'val_b'}
# 注意这里的数字 0 代表引用的是 format 中的第一个对象
>>> b = a
>>> 'X: {0[a]};  Y: {1[b]}'.format(a, b)
'X: val_a;  Y: val_b'
```
2. 数值格式转换
* ‘b’ - 二进制。将数字以2为基数进行输出。
* ‘c’ - 字符。在打印之前将整数转换成对应的Unicode字符串。
* ‘d’ - 十进制整数。将数字以10为基数进行输出。
* ‘o’ - 八进制。将数字以8为基数进行输出。
* ‘x’ - 十六进制。将数字以16为基数进行输出，9以上的位数用小写字母。
* ‘e’ - 幂符号。用科学计数法打印数字。用’e’表示幂。
* ‘g’ - 一般格式。将数值以fixed-point格式输出。当数值特别大的时候，用幂形式打印。
* ‘n’ - 数字。当值为整数时和’d’相同，值为浮点数时和’g’相同。不同的是它会根据区域设置插入数字分隔符。
* ‘%’ - 百分数。将数值乘以100然后以fixed-point(‘f’)格式打印，值后面会有一个百分号。
```py
# 整数格式化
>>> print('{0:b}'.format(3))
11
>>> print('{:c}'.format(97))
a
>>> print('{:d}'.format(20))
20
>>> print('{:o}'.format(20))
24
>>> print('{:x},{:X}'.format(0xab, 0xab))
ab,AB
# 浮点数格式化
>>> print('{:e}'.format(20))
2.000000e+01
>>> print('{:g}'.format(20.1))
20.1
>>> print('{:f}'.format(20))
20.000000
>>> print('{:n}'.format(20))
20
>>> print('{:%}'.format(20))
2000.000000%
```

我经常用的格式化输出方法示例：
```py
print(format(interest,".2f"))
```


#### 多重输入的解决方法示例
```py
def count_ways(n):
    if n == 1:
        return 1
    elif n == 2:
        return 2
    else:
        ways = [0] * (n + 1)
        ways[1] = 1
        ways[2] = 2
        for i in range(3, n + 1):
            ways[i] = ways[i - 1] + ways[i - 2]
        return ways[n]

while True:
    try:
        n = int(input())
        result = count_ways(n)
        print(result)
    except EOFError:
        break
```
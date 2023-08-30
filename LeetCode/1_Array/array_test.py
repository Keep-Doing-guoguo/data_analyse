class ArrayTest:
    def test(self):
        a = []
        #添加元素
        a.append(1)
        a.append(2)
        a.append(3)
        print(a)

        #插入元素
        a.insert(2,99)
        print(a)

        #查找元素
        temp = a[2]
        print(temp)

        #修改元素
        a[2] = 88
        print(a)

        #删除元素
        a.remove(88)
        print(a)
        a.pop()
        print(a)

        #寻找一个元素
        index = a.index(2)
        print(index)
test = ArrayTest()
test.test()
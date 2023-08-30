class Test:
    def test(self):
        #创建一个set
        s = set()

        #增加元素
        s.add(10)
        s.add(3)
        s.add(2)
        s.add(2)
        s.add(1)
        print(s)

        #检查是否存在
        print(2 in s)

        #删除元素
        s.remove(2)
        print(s)
Test().test()
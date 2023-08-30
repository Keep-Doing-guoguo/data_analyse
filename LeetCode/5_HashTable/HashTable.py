class Test:
    def test(self):
        hashtable = ['']*4#使用空列表创建的哈希表
        mapping = {}#字典创建的哈希表

        #增加元素
        hashtable[1] = 'hanmeimei'
        hashtable[2] = 'lihua'
        hashtable[3] = 'siyangyuan'
        mapping[1] = 'hanmeimei'
        mapping[2] = 'lihua'
        mapping[3] = 'siyangyuan'
        print(hashtable)
        print(mapping)
        print(1 in mapping.keys())
        #xiugai修改元素
        hashtable[1] = 'bishi'
        mapping[1] = 'bishi'
        print(hashtable)
        print(mapping)

        #删除元素
        hashtable[1] = ''
        mapping.pop(1)
        print(hashtable)
        print(mapping)

        #fifnd元素
        hashtable[3]
        mapping[3]

        #jian检查元素
        3 in mapping
Test().test()



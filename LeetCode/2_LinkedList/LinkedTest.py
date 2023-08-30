from collections import deque
'''
通常链表的删除和插入的时间复杂度是O(1),查找和访问和搜索是O(n)
'''
class Test:
    def test(self):
        linkedlist = deque()

        #创建链表
        linkedlist.append(1)
        linkedlist.append(2)
        linkedlist.append(3)
        print(linkedlist)

        #插入元素
        linkedlist.insert(2,99)
        print(linkedlist)

        #获取元素
        element = linkedlist[2]
        print(element)

        #查找元素的索引
        index = linkedlist.index(99)
        print(index)

        #修改元素的值
        linkedlist[2] = 88
        print(linkedlist)

        #删除链表的值
        linkedlist.remove(88)
        print(linkedlist)


t1 = Test()
t1.test()
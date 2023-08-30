# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution(object):
    def removeElements(self, head, val):
        """
        :type head: ListNode
        :type val: int
        :rtype: ListNode
        链表三部曲：首先要定一个一个dummy节点，来指向head节点。其次再定义一个pre节点，来指向dummy节点。
        pre节点将会比head节点慢一步。
        """
        #定义一个空节点
        dummy = ListNode(0)
        dummy.next = head
        prev = dummy
        while head is not None:
            if head.val == val:
                prev.next = head.next
            else:
                prev = prev.next
            head = head.next#head是会始终向下一个位置进行移动的
        return dummy.next#最终返回的是dummy的下一个节点。


num1 = ListNode(1)
num2 = ListNode(2)
num3 = ListNode(6)
num4 = ListNode(3)
num5 = ListNode(4)
num6 = ListNode(5)
num7 = ListNode(6)
num1.next = num2
num2.next = num3
num3.next = num4
num4.next = num5
num5.next = num6
num6.next = num7
Solution().removeElements(num1,6)

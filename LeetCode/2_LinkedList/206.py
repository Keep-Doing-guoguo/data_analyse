# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution(object):
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        dummy = ListNode(0)
        dummy.next = head

        while head is not None and head.next is not None:
            dnext = dummy.next
            hnext = head.next

            dummy.next = hnext
            head.next = hnext.next
            hnext.next = dnext
        return dummy.next

num1 = ListNode(1)
num2 = ListNode(2)
num3 = ListNode(3)
num4 = ListNode(4)
num5 = ListNode(5)
num6 = ListNode(6)
num7 = ListNode(6)
num1.next = num2
num2.next = num3
num3.next = num4
num4.next = num5
num5.next = num6
num6.next = num7
Solution().reverseList(num1)
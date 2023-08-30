# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        total = 0
        nextl = 0
        result = ListNode()
        cur = result
        while (l1 != None and l2 != None):
            total = l1.val + l2.val + nextl
            cur.next = ListNode(total%10)
            nextl = total/10
            l1 = l1.next
            l2 = l2.next
            cur = cur.next


# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
from collections import deque
from queue import Queue
class Solution(object):
    def levelOrderBottom(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        result = []
        if root == None:
            return result

        q = deque()
        q.append(root)#默认从右边加入
        temp = deque([])
        while len(q) > 0:
            size = len(q)
            ls =[]
            while size > 0:
                cur = q.popleft()
                ls.append(cur.val)
                if cur.left != None:
                    q.append(cur.left)
                if cur.right != None:
                    q.append(cur.right)
                size = size - 1
            temp.appendleft(ls[:])
        result = list(temp)#数据类型的转换
        return temp
node1 = TreeNode(3)
node2 = TreeNode(9)
node3 = TreeNode(20)
node4 = TreeNode(15)
node5 = TreeNode(7)

node1.left = node2
node1.right = node3
node2.left = node4
node2.right = node5
print(Solution().levelOrderBottom(node1))
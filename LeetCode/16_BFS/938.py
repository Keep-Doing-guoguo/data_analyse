# Definition for a binary tree node.
from queue import Queue
class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution(object):
    def rangeSumBST(self, root, low, high):
        """
        :type root: TreeNode
        :type low: int
        :type high: int
        :rtype: int
        """
        #当想要使用BFS的时候，一定要想到队列Queue。
        result = 0
        q = Queue()
        q.put(root)#写入
        while (q.qsize() > 0):
            size = q.qsize()#获取到这一层中的元素个数
            while size > 0:
                cur = q.get()#写出
                if low<=cur.val<=high:
                    result = result + cur.val
                if cur.left != None:
                    q.put(cur.left)
                if cur.right != None:
                    q.put(cur.right)
                size = size-1
        return result

node1 = TreeNode(10)
node2 = TreeNode(5)
node3 = TreeNode(15)
node4 = TreeNode(3)
node5 = TreeNode(7)
node6 = TreeNode(18)
node7 = TreeNode(0)
node1.left = node2
node2.left = node4
node2.right = node5
node1.right = node3
node3.right = node6
print(Solution().rangeSumBST(node1,7,15))
# Definition for a binary tree node.
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
        if root == None:
            return 0

        leftsum = self.rangeSumBST(root.left,low,high)
        rightsum = self.rangeSumBST(root.right,low,high)

        result = leftsum + rightsum
        if (low <= root.val) and (root.val <= high):
            result = result + root.val
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

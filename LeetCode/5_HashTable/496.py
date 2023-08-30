class Solution(object):
    def nextGreaterElement(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """

        stack = []
        ht = {}
        res = []
        for num in nums2:
            # 栈不为空，且num始终还得大于栈顶元素，才可以取出来，放入到table里面
            while len(stack) != 0 and num > stack[-1]:
                temp = stack.pop()
                ht[temp] = num
            stack.append(num)

        while len(stack) != 0:
            ht[stack.pop()] = -1
        for num in nums1:
            res.append(ht[num])
        return res
nums1 = [8,1,2]
nums2 = [2,1,0,8,7,6,5]
Solution().nextGreaterElement(nums1,nums2)
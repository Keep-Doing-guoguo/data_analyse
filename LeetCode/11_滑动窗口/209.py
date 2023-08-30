class Solution(object):
    def minSubArrayLen(self, target, nums):
        """
        :type target: int
        :type nums: List[int]
        :rtype: int
        当和小于target的时候，将窗口向右移动，既扩大窗口。当和大于target的时候，将窗口中的左元素剔除。
        """
        #适用于定长性的问题
        #边界条件一定要标明：
        if nums is None or len(nums) == 0:
            return 0

        res = len(nums) + 1
        total = 0
        i = 0
        j = 0
        while j < len(nums):
            total = total + nums[j]
            j = j + 1
            while total >= target:
                res = min(res,j-i)
                total = total - nums[i]
                i = i + 1
        if res == len(nums):
            return 0
        else:
            return res
target = 7
nums = [2,3,1,2,4,3]
Solution().minSubArrayLen(target,nums)
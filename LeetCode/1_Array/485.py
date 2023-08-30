class Solution(object):
    def findMaxConsecutiveOnes(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if nums is None or len(nums) == 0:
            return 0

        count = 0
        result = 0
        for i in range(len(nums)):
            if nums[i] == 1:
                count+=1
            else:
                result = max(count,result)
                count = 0
        return max(result,count)

class Solution(object):
    def searchInsert(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        if nums == None or len(nums) == 0:
            return 0
        l = 0
        r = len(nums) - 1
        while(l < r):
            m = l + (r-l)//2
            if nums[m] == target:
                return m
            elif nums[m] >target:
                r = m
            else:
                l = m + 1
        if nums[l] > target:
            return l
        else:#这里是不可能相等的，因为相等的话，再上面就已经返回了。
            return l+1
nums = [1,3,5,6]
target = 7
Solution().searchInsert(nums,target)
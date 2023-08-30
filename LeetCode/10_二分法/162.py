class Solution(object):
    def findPeakElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        只要左边的数字大于右边的数字，那么就会存在左边会有一个峰值存在，此时移动r即可
        只要右边的数字大于左边的数字，那么就会存在右边会有一个峰值存在，此时移动l即可

        左边的数字大于右边的数字，向左移动r
        右边的数字大于左边的数字，向右移动l

        O（logn）的时候就应该想到二分查找法
        """
        if nums is None or len(nums) == 0:
            return -1
        l = 0
        r = len(nums) - 1
        while (l < r):
            mid = l + (r - l) // 2
            if nums[mid] > nums[mid + 1]:
                r = mid
            else:
                l = mid + 1
        return l

nums = [1,2,1,3,5,6,4]
Solution().findPeakElement(nums)
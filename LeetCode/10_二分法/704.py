class Solution(object):
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        if nums == None or len(nums) == 0:
            return -1
        left,right = 0,len(nums)-1
        while(left <= right):
            mid = left + (right - left)//2#有可能会超过数组的边界值
            if nums[mid] == target:
                return mid
            elif nums[mid] > target:
                right = mid - 1
            else:
                left = mid + 1
        return -1
nums = [-1,0,3,5,9,12]
target = 9
Solution().search(nums,target)
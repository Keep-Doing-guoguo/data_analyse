class Solution(object):
    def removeElement(self, nums, val):
        """
        :type nums: List[int]
        :type val: int
        :rtype: int
        双指针解法
        该题的本意旨在最后将所有等于val的地方，移动到数组的最后面。将不等于val的值移动到数组的最前面，使用的数组的交换方式。
        """
        if nums == None or len(nums) == 0:
            return 0
        l = 0
        r = len(nums) - 1
        while(l < r):
            while(l < r and nums[l] != val):
                l = l+1
            while(l < r and nums[r] == val):
                r = r-1
            nums[l] ,nums[r] = nums[r] ,nums[l]

        return l if nums[l] == val else l+1

#测试元素交换
nums1 = [0,1,2,2,3,0,4,2]
#nums1[0], nums1[2] = nums1[2], nums1[0]
print(Solution().removeElement(nums1,2))
class Solution:
    def searchrange(self,nums,target):
        leftindex = self.binarysearch(nums,target)

        #如果左边界不存在，或者没有查询到该值
        if leftindex>=len(nums) or nums[leftindex] != target:
            return [-1,-1]

        pass

    def binarysearch(self,nums,target):
        left = 0
        right = len(nums) - 1
        while left <= right:
            mid = left+(right-left)/2
            if nums[mid]>=target:
                right=mid-1
            else:
                left=mid+1
        return left

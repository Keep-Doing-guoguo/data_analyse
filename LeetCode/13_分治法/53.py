class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        return self.getMax(nums,0,len(nums)-1)
    def getMax(self,nums,l,r):
        if l == r:
            return nums[l]
        mid = l + (r-l)//2
        leftSum = self.getMax(nums,l,mid)
        rightSum = self.getMax(nums,mid+1,r)
        crossSum = self.crossSum(nums,l,r)
        return max(leftSum,rightSum,crossSum)

    def crossSum(self,nums,l,r):
        mid = l + (r-l)//2
        leftSum = nums[mid]
        leftMax = leftSum
        for i in range(mid-1,l-1,-1):
            leftSum = leftSum + nums[i]
            leftMax = max(leftMax,leftSum)
        rightSum = nums[mid+1]
        rightMax = rightSum
        for i in range(mid+2,r+1):
            rightSum = rightSum + nums[i]
            rightMax = max(rightMax,rightSum)
        return leftMax+rightMax
        pass
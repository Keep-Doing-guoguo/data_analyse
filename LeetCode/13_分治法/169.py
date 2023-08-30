class Solution():
    def majorityElement(self,nums):
        return self.getMajority(nums,0,len(nums)-1)
    def getMajority(self,nums,left,right):
        if left == right:
            return nums[left]

        mid = left + (right-left)//2
        #mid = (right+left)/2
        leftmajority = self.getMajority(nums,left,mid)
        rightmajority = self.getMajority(nums,mid+1,right)

        if leftmajority == rightmajority:
            return leftmajority

        leftcount = 0
        rightcount = 0
        for i in range(left,right+1):
            if nums[i] == leftmajority:
                leftcount += 1
            elif nums[i] == rightmajority:
                rightcount += 1
        #这个是随机返回的。
        return leftmajority if leftcount > rightcount else rightmajority
nums = [2,2,3,1,2,2,3]
Solution().majorityElement(nums)
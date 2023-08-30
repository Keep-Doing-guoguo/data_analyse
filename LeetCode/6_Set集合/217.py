class Solution:
    def containsDuplicate(self, nums: list[int]) -> bool:
        return len(set(nums)) != len(nums)

solu = Solution()
nums = [1,2,3,1]
#print(set(nums))
print(solu.containsDuplicate(nums))

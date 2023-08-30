# 第一种方法
# class Solution(object):
#     def subsets(self, nums):
#         """
#         :type nums: List[int]
#         :rtype: List[List[int]]
#         """
#         '''
#         使用的是扩展法，一个一个数的向上扩展，向上加。
#         '''
#         result = []
#         result.append([])
#         for num in nums:
#             temp = []
#             for res in result:
#                 r = res.copy()#新建一个变量，复制res的值，防止引用传递。
#                 r.append(num)
#                 temp.append(r)
#             for t in temp:
#                 result.append(t)
#
#         return result
#
# 第一种方法

# # 第二种方法
# class Solution(object):
#     def subsets(self, nums):
#         """
#         :type nums: List[int]
#         :rtype: List[List[int]]
#         """
#         result = []
#         result.append([])
#         for i in range(1,len(nums)+1):
#             self.backtracking(nums,result,i,0,[])
#         return result
#     def backtracking(self,nums,result,length,index,subset):
#         if len(subset) == length:
#             temp = subset.copy()
#             result.append(temp)
#             return
#         for i in range(index,len(nums)):
#             subset.append(nums[i])
#             self.backtracking(nums,result,length,i+1,subset)
#             subset.pop()
# # 第二种方法

# 第三种方法




#第三种方法
class Solution(object):
    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        result = []
        self.dfs(nums,result,0,[])
        return result
    def dfs(self,nums,result,index,subset):
        result.append(subset.copy())
        if index == len(nums):
            return
        for i in range(index,len(nums)):
            subset.append(nums[i])
            self.dfs(nums,result,i+1,subset)
            subset.pop()

nums = [1,2,3]
print(Solution().subsets(nums))


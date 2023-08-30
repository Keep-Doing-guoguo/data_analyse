class Solution:
    def subsets(self,nums):
        result = []
        self.dfs(nums,result,0,[])
        return result
    def dfs(self,num,result,index,subset):

        pass
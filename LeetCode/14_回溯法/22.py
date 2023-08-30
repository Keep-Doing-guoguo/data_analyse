class Solution():
    #['(())', '()()']
    '''
    n = 2

    '''
    def generateParenthesis(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        result = []
        self.backtracking(n,result,0,0,'')
        return result
    def backtracking(self,n,result,left,right,str):
        if right > left:
            return
        if (left == right) and (right == n):
            result.append(str)
            return
        if left < n:#加左括号。当左括号小于n的时候，加左括号。n代表的是总括号数。
            self.backtracking(n,result,left+1,right,str+'(')
        if right < left:#当右括号小于左括号的时候，需要加右括号。
            self.backtracking(n,result,left,right+1,str+')')


n = 2
re = Solution().generateParenthesis(n)
print(re)
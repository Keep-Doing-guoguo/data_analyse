class Solution(object):
    def maxVowels(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: int
        """
        if s == None or len(s) == 0 or k > len(s):
            return 0
        hashset = ('a','o','e','i','u')
        res = 0
        count = 0
        #这是第一个0-k
        for i in range(0,k):
            if s[i] in hashset:
                count = count + 1
        res = max(res,count)
        #只需要检查新加进来的和踢出去的元素。
        for i in range(k,len(s)):
            out = s[i-k]
            inc = s[i]
            if out in hashset:
                count = count - 1
            if inc in hashset:
                count = count + 1
            res = max(res,count)
        return res
s = "leetcode"
k = 3
Solution().maxVowels(s,k)
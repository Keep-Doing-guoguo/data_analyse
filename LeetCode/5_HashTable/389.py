class Solution(object):
    def findTheDifference(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        '''
        key:存放ascll码
        value：存放次数 
        '''
        if len(s) == 0:
            return t
        #创建一个数组
        table = [0]*26
        for i in range(len(t)):
            if i < len(s):
                table[ord(s[i])-ord('a')] -= 1
            table[ord(t[i])-ord('a')] += 1
        for i in range(26):
            if table[i] != 0:
                return chr(i+97)
a = 'abcd'
t = 'dcbae'
Solution().findTheDifference(a,t)
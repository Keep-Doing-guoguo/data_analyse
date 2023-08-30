class Solution(object):
    def uniquePaths(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
        dp = [([0]*n) for _ in range(m)]
        dp[0][0] = 1

        for i in range(0,m):
            for j in range(0,n):
                if i-1 >= 0 and i - 1 < m:
                    dp[i][j] = dp[i][j] + dp[i-1][j]
                if j-1 >= 0 and j - 1 < n:
                    dp[i][j] = dp[i][j] + dp[i][j-1]
        #print(dp)
        return dp[m-1][n-1]

Solution().uniquePaths(3,7)
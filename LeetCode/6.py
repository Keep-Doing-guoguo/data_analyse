class Solution:
    def convert(self, s: str, numRows: int) -> str:
        if numRows < 2:
            return s
        res = ["" for _ in range(numRows)]
        i, flag = 0, -1
        for c in s:
            res[i] = res[i] + c
            #if i == 0 or i == numRows - 1:
            if i == 0 or i == 2:
                flag = -flag
            i = i + flag
        return "".join(res)

s = "LEETCOD"
numRows = 3
Solution().convert(s,numRows)
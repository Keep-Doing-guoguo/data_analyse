class Solution(object):
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        就是一个二维数组的索引和一位数组的索引进行相互转换的过程
        一位转换为二维需要：模列数，除以列数
        """
        #
        row = len(matrix)
        col = len(matrix[0])
        l = 0
        r = row * col - 1
        while l <= r:
            m = l+(r-l)//2
            element = matrix[m//col][m%col]
            if element == target:
                return True
            elif element > target:
                r = m - 1
            else:
                l = m + 1
        return False

matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]]
target = 3
Solution().searchMatrix(matrix,target)
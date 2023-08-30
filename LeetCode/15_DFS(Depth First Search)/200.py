class Solution(object):
    def numIslands(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """
        if grid == None or len(grid) == 0:
            return 0

        result = 0
        row = len(grid)
        col = len(grid[0])
        for i in range(0,row):
            for j in range(0,col):
                if grid[i][j] == '1':
                    result = result + 1
                    self.dfs(grid,i,j,row,col)
        return result
    def dfs(self,grid,x,y,row,col):
        if x < 0 or y < 0 or x >= row or y >= col or grid[x][y] == '0':
            return
        grid[x][y] = '0'
        self.dfs(grid,x+1,y,row,col)
        self.dfs(grid,x-1,y,row,col)
        self.dfs(grid,x,y-1,row,col)
        self.dfs(grid,x,y+1,row,col)


grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
print(Solution().numIslands(grid))
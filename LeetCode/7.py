class Solution(object):

    def reverse(self, x: int) -> int:
        if -10 < x < 10:
            return x
        str_x = str(x)
        if str_x[0] != "-":#进到这里面是正数
            str_x = str_x[::-1]
            x = int(str_x)
        else:#进到这里面是负数
            str_x = str_x[:0:-1]
            x = int(str_x)
            x = -x
        return x if -2147483648 < x < 2147483647 else 0

x = -123
print(Solution().reverse(x))
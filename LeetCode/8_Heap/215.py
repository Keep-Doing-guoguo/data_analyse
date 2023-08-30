import heapq
class Solution(object):
    def findKthLargest(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        #默认创建的是一个最小堆，所以需要进行乘-1进行反转。
        heap = []#创建一个heap，堆操作。
        heapq.heapify(heap)
        for num in nums:
            heapq.heappush(heap,num*-1)#add操作

        while k > 1:
            heapq.heappop(heap)#直接进行出栈操作。加入k=2，那么将会进行出栈一次操作。
            k = k-1
        return heapq.heappop(heap)*-1

nums = [3,2,1,5,6,4]
k = 2
Solution().findKthLargest(nums,k)
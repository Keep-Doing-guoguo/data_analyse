import heapq


class Solution(object):
    def topKFrequent(self, words, k):
        """
        :type words: List[str]
        :type k: int
        :rtype: List[str]
        """
        '''
        使用的是hashtable，字典操作。
        使用的是heap和hashtable联合操作。
        '''
        mapping = {}
        for word in words:
            if word not in mapping:
                mapping[word] = 0
            mapping[word] = mapping[word]+1
        print(mapping)


        heap = []
        for key,value in mapping.items():
            heapq.heappush(heap,Node(key,value))
            if len(heap) > k:
                heapq.heappop(heap)#出栈操作

        res = []
        while len(heap) > 0:
            temp = heapq.heappop(heap)
            print(temp.key,' ',temp.value)
            res.append(temp.key)
        res.reverse()

        return res

class Node():
    def __init__(self,key,value):
        self.key = key
        self.value = value

    #使用<操作符号,返回的是一个true或者是false,将大于号进行了重新的定义
    '''
    if self.value == nxt.value:
        return: self.key > nxt.key
    else:
        return: self.value < nxt.value
    '''
    #a = 97
    #i > l
    #value相等就继续比较value，value不相等就比较key。
    #在这里是用来将heap用来排序使用的。最小堆操作。
    def __lt__(self, nxt):
        #return self.key > nxt.key if self.value == nxt.value else self.value < nxt.value
        if self.value == nxt.value:
            return self.key > nxt.key
        else:
            return self.value < nxt.value

words = ["i", "love", "leetcode", "i", "love", "coding"]
k = 2
Solution().topKFrequent(words,k)
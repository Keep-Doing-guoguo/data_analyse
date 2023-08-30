class Solution(object):
    def containsDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        哈希表的用处：
        1：用数组创建的哈希表。明确数组里有多少元素。
        2：用自带函数创建的哈希表。dict
        """
        '''
        key：数组中的元素
        value：数组中元素出现的次数
        '''
        if len(nums) == 0:
            return False
        hashtable = {}
        for num in nums:
            if num not in hashtable:
                hashtable[num] = 1
            else:
                temp = hashtable.get(num)
                hashtable[num] += 1
        for key in hashtable.keys():
            if hashtable.get(key) > 1:
                return True
        return False




class Solution(object):
    def containsDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        mapping = {}
        if len(nums) == 0:
            return False
        for num in nums:
            if num not in mapping:
                mapping[num] = 1
            else:
                mapping[num] = mapping.get(num)+1
        for v in mapping.values():
            if v>1:
                return True
        return False
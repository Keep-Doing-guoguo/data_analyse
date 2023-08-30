class MyHashSet(object):

    def __init__(self):
        #缺点就是数组很大。前提是数组大小已知。
        self.hashset = [0]*1000001

    def add(self, key):
        """
        :type key: int
        :rtype: None
        """
        self.hashset[key] = 1

    def remove(self, key):
        """
        :type key: int
        :rtype: None
        """
        self.hashset[key] = 0

    def contains(self, key):
        """
        :type key: int
        :rtype: bool
        """
        return  bool(self.hashset[key])


# Your MyHashSet object will be instantiated and called as such:
# obj = MyHashSet()
# obj.add(key)
# obj.remove(key)
# param_3 = obj.contains(key)
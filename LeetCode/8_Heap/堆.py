import heapq
class Test:
    def test(self):
        #创建minheap
        minheap = []
        heapq.heapify(minheap)

        #add 元素
        heapq.heappush(minheap,10)
        heapq.heappush(minheap, 8)
        heapq.heappush(minheap, 9)
        heapq.heappush(minheap, 2)
        heapq.heappush(minheap, 1)
        heapq.heappush(minheap, 11)
        #[1, 2, 9, 10, 8, 11]
        print(minheap)

        #出栈，栈顶元素
        print(minheap[0])

        #delete
        heapq.heappop(minheap)

        #size
        len(minheap)

        #迭代
        while len(minheap) != 0:
            print(heapq.heappop(minheap))

if __name__ == '__main__':
    test = Test()
    test.test()
    print(10*-1)
    print(True if 'i'>'love' else False)
    print(2 > 3 if 2 == 2 else 2 < 1)
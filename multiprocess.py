from multiprocessing import Pool, Process
import time
import math

N = 5000000

def cube(x):
    return math.sqrt(x)

if __name__ == "__main__":
    with Pool() as pool:
      result = pool.map(cube, range(10,N))
    print("Program finished!")

from multiprocessing import Process

def bubble_sort(array):
    check = True
    while check == True:
      check = False
      for i in range(0, len(array)-1):
        if array[i] > array[i+1]:
          check = True
          temp = array[i]
          array[i] = array[i+1]
          array[i+1] = temp
    print("Array sorted: ", array)

if __name__ == '__main__':
    p = Process(target=bubble_sort, args=([1,9,4,5,2,6,8,4],))
    p2 = Process(target=bubble_sort, args=([1,9,4,5,2,6,8,4],))
    p3 = Process(target=bubble_sort, args=([1,9,4,5,2,6,8,4],))
    p.start()
    p2.start()
    p3.start()
    p.join()
    p2.join()
    p3.join()

    with Pool() as pool:
      for i in range(N):
        #result = pool.map(bubble_sort, [1,9,4,5,2,6,8,4])
        result = pool.map(cube, range(10,N))
    print("Program finished!")


# p1 = mp.Process(collect_data())
# p2 = mp.Process(dashboard())

# p1.start()
# p2.start()

# p1.join()
# p2.join()

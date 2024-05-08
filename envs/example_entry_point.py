import time
import sys
import random
if __name__=="__main__":
    with open("./log.txt",'w'):
        # print("starting an exp")
        print(time.time())
        time.sleep(random.random())
        # print("finish an exp")
        sys.exit()
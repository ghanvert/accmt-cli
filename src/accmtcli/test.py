import os

try:
    os.system("python test2.py")
except KeyboardInterrupt:
    print("detected keyboard interrupt")

print("finished")
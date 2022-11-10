#import fire
import sys
from solarnet.run import RunTask
method = sys.argv[1]

if __name__ == '__main__':
    #fire.Fire(RunTask)
    if method=="make_masks":
        RunTask.make_masks()
    elif method=="split_images":
        RunTask.split_images()
    elif method=="train_segmenter":
        RunTask.train_segmenter()
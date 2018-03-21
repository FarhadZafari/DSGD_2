import sys
import os

print(os.environ.get('PYTHONPATH', ''))

print(sys.path)


sample = set()
sample.add('1')

print(sample)
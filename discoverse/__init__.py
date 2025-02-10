import os

print(os.path.realpath(__file__))

DISCOVERSE_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DISCOVERSE_ASSERT_DIR = os.path.join(DISCOVERSE_ROOT_DIR, 'models')



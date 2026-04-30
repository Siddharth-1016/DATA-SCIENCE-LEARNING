import logging

logging.basicConfig(
    filename='app.log',
    filemode='w',
    level=logging.DEBUG,
    format='%(asctime)s-%(name)s-%(levelname)s-%(message)s', # Change 1: Fixed attribute name
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True  # Change 2: Force reconfiguration
)
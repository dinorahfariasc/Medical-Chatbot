import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: - %(message)s:')

listFiles = [
    'src/__init__.py',
    'src/helper.py',
    'src/prompt.py',
    '.env',
    'setup.py',
    'app.py',
    'research/trials.ipynb'
] 

for filepath in listFiles:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != '':
        os.makedirs(filedir, exist_ok=True)
        logging.info(f'Creating directory: {filedir} for the file: {filename}')

    if(not os.path.exists(filepath) or (os.path.getsize(filepath) == 0)):
        with open(filepath, 'w') as file:
            logging.info(f'Creating empty file: {filename}')
    
    else:
        logging.info(f'File: {filename} already exists')
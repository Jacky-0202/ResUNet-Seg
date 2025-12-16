# utils/logger.py

import csv
import os

class CSVLogger:
    def __init__(self, save_dir, filename='training_log.csv'):
        """
        Args:
            save_dir (str): Directory where the log file will be saved.
            filename (str): Name of the CSV file.
        """
        self.save_dir = save_dir
        self.filepath = os.path.join(save_dir, filename)
        
        # [Fix] Define headers here to keep train.py clean
        self.headers = ['Epoch', 'LR', 'Train_Loss', 'Train_Dice', 'Val_Loss', 'Val_Dice']
        
        # Ensure the directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        # Create file and write headers (overwrite mode 'w')
        with open(self.filepath, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)
        
        print(f"📝 Log file created at: {self.filepath}")

    def log(self, data):
        """
        Write a single row of data to the CSV.
        Args:
            data (list): List of data values to write.
        """
        with open(self.filepath, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(data)
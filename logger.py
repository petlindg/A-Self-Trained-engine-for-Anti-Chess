from colorama import *
from datetime import datetime
from pathlib import Path
import platform
from os.path import exists
import os

init(autoreset=True)

def singleton(class_):
    instances = {}
    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance

@singleton
class Logger:
    
    log_colors = {
    'datetime': {'text_color': Fore.CYAN, 'bg_color': Back.RESET},
    'info': {'text_color': Fore.WHITE, 'bg_color': Back.BLUE},
    'exception': {'text_color': Fore.WHITE, 'bg_color': Back.LIGHTYELLOW_EX}, 
    'error': {'text_color': Fore.WHITE, 'bg_color': Back.RED},
    'warning': {'text_color': Fore.WHITE, 'bg_color': Back.YELLOW},
    'critical': {'text_color': Fore.WHITE, 'bg_color': Back.MAGENTA},  
    }

        
    def __init__(self, identifier):
        self.task_name = identifier
        self.timestamp_format = '%d/%m/%Y %H:%M:%S.%f'
        self.log_format = '%DATETIME% || %CATEGORY% || %MESSAGE%'

        self.directory_path = ''
        self.log_filename = ''

        self.get_directory_path()
        self.get_log_filename()


    def set_log_format(self, log_format):
        self.log_format = log_format


    def get_log_format(self):
        return self.log_format


    def current_time(self):
        now = datetime.now()
        return now.strftime(self.get_timestamp_format())


    def set_timestamp_format(self, format):
        self.timestamp_format = format


    def get_timestamp_format(self):
        return self.timestamp_format


    def ensure_directory_exists(self):
        if not exists(self.directory_path):
            try:
                os.makedirs(self.directory_path)
            except:
                print('Error creating log folder.')


    def log_file_exists(self):
        path = Path(f'{self.directory_path}/{self.log_filename}')
        return path.is_file()


    def get_directory_path(self):
        self.directory_path = Path(f'./logs/{self.task_name}/{platform.system()}')
        self.directory_path.mkdir(parents=True, exist_ok=True)


    def get_log_filename(self):
        self.ensure_directory_exists()
        current_dt = datetime.now()
        month = str(current_dt.month).zfill(2)

       
        self.log_filename = f'{self.task_name}_{current_dt.day}{month}{current_dt.year}.log'
        
        if not self.log_file_exists():
            try:
                self.add_new_log()
            except:
                print('Error creating log file.')


    def add_new_log(self):
        self.ensure_directory_exists()
        current_dt = datetime.now()
        month = str(current_dt.month).zfill(2)

        header = (
            f'\n\n{"=" * 100}\n'
            f'{"=" * 100}\n'
            f'{"=" * 10}{" " * 80}{"=" * 10}\n'
            f'{"=" * 10}{" " * 30}Welcome to the log file{" " * 27}{"=" * 10}\n'
            f'{"=" * 10}{" " * 80}{"=" * 10}\n'
            f'{"=" * 10}{" " * 80}{"=" * 10}\n'
            f'{"=" * 10}{" " * 30}Initialized on: {current_dt.day}/{month}/{current_dt.year}{" " * 24}{"=" * 10}\n'
            f'{"=" * 10}{" " * 80}{"=" * 10}\n'
            f'{"=" * 10}{" " * 80}{"=" * 10}\n'
            f'{"=" * 100}\n'
            f'{"=" * 100}\n\n\n'
        )
        file_path = Path(f'{self.directory_path}/{self.log_filename}')
        with file_path.open('a') as f:
            f.write(header)

        self.info(f'Log file has been created successfully for "{self.task_name}".')


    def add_new_record(self, new_reg):
        if not self.log_file_exists():
            self.add_new_log()

        with open(f'{self.directory_path}/{self.log_filename}', 'a', encoding='utf-8') as f:
            f.write(f'{new_reg}\n')


    def info(self, message):
        msg = self.format_message(self.log_format, 'info', message)
        self.add_new_record(msg)


    def exception(self, message):
        msg = self.format_message(self.log_format, 'exception', message)
        self.add_new_record(msg)


    def error(self, message):
        msg = self.format_message(self.log_format, 'error', message)
        self.add_new_record(msg)


    def warning(self, message):
        msg = self.format_message(self.log_format, 'warning', message)
        self.add_new_record(msg)


    def critical(self, message):
        msg = self.format_message(self.log_format, 'critical', message)
        self.add_new_record(msg)


    def format_message(self, string, category, message):
        sections = {
            'DATETIME': str(self.current_time()),
            'CATEGORY': self.format_level(category),
            'MESSAGE': self.format_section(message)
        }
        for section in sections:
            string = string.replace(f'%{section}%', sections[section])
        return string


    def format_level(self, category):
        if category in self.log_colors:
            text_color = self.log_colors[category]['text_color']
            bg_color = self.log_colors[category]['bg_color']
           
            colored_category = f'{text_color}{bg_color}[{category.upper()}]{Style.RESET_ALL}'
            return colored_category
        else:
            return ''


    def format_section(self, section, length=None):
        section = str(section)
        if length is not None and len(section) > length:
            section = section[:length-3] + '...'
        return section
    


    
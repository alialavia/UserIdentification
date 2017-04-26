import csv
import datetime
import time

class BGColor:
    black = '40m '
    red = '41m '
    green = '42m '
    yellow = '43m '
    blue = '44m '
    purple = '45m '
    cyan = '46m '
    white = '47m '


class TextStyle:
    default = '0'
    bold = '1'
    underline = '2'
    negative1 = '3'
    negative2 = '5'


class FontColor:
    black = '30'
    red = '31'
    green = '32'
    yellow = '33'
    blue = '34'
    purple = '35'
    cyan = '36'
    white = '37'

    # background transformers

    # end color
    ENDC = '\033[0m'


class Logger:

    """
    Todo: Add logging to harddrive
    """

    # log files
    log_to_file = False
    filemode = 'ab'
    filename = 'debug.log'

    # output format \033[ text_style text_color bg_color
    endc = '\033[0m'
    thresh = 0
    levels = {
        0: 'all',
        1: 'info',       # white
        2: 'debug',  # white
        3: 'warning',   # bg yellow
        4: 'error',      # bg red
        5: 'severe'      # bg red
    }
    scope_styles = {
        'cl': FontColor.blue + ";" + BGColor.black,
        'db': FontColor.green + ";" + BGColor.black,
        'server': FontColor.black + ";" + BGColor.white,
        'cnn': FontColor.purple + ";" + BGColor.black
    }

    def __init__(self):
        pass

    @classmethod
    def Config(self, dump_log=False, filemode='ab', filename=None):
        if filemode not in {'ab', 'wb'}:
            print "Logger Config: Please choose between modes: {ab, wb}"
        self.log_to_file = dump_log
        self.filemode = filemode
        if filename is not None:
            self.filename = filename

    @classmethod
    def write_log(self, scope='', msg='', level=''):
        if self.log_to_file is False:
            return
        try:
            with open(self.filename, self.filemode) as f:
                writer = csv.writer(f, delimiter=',')
                timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
                writer.writerow([timestamp, scope, level, msg])
        except:
            print "\033[0;" + FontColor.black + ";" + BGColor.red + "Could not write log file" + self.endc

    @classmethod
    def severe(self, msg, scope=None):
        print "\033[0;" + FontColor.black + ";" + BGColor.red + msg + self.endc
        self.write_log(scope, msg, 'severe')

    @classmethod
    def error(self, msg, scope=None):
        print "\033[0;" + FontColor.white + ";" + BGColor.red + msg + self.endc
        self.write_log(scope, msg, 'error')

    @classmethod
    def warning(self, msg, scope=None):
        if self.thresh < 4:
            print "\033[0;" + FontColor.black + ";" + BGColor.yellow + msg + self.endc
            self.write_log(scope, msg, 'warning')

    @classmethod
    def debug(self, scope, msg):
        if self.thresh < 3:
            self.print_msg(scope, msg)
            self.write_log(scope, msg, 'debug')

    @classmethod
    def info(self, scope, msg):
        if self.thresh < 2:
            self.print_msg(scope, msg)
            self.write_log(scope, msg, 'info')

    @classmethod
    def print_msg(self, scope, msg):
        if scope not in self.scope_styles:
            style = FontColor.white + ";" + BGColor.black
        else:
            style = self.scope_styles[scope]
        print "\033[0;" + style + msg + " " + self.endc

    @classmethod
    def print_clr(self, msg, color="white", bg="red"):
        print "\033[0;" + getattr(FontColor, color) + ";" + getattr(BGColor, bg) + msg + self.endc

    @classmethod
    def print_example(self):
        print("\033[1;37;40m \033[2;37:40m TextColour BlackBackground          TextColour GreyBackground                WhiteText ColouredBackground\033[0;37;40m\n")
        print("\033[1;30;40m Dark Gray      \033[0m 1;30;40m            \033[0;30;47m Black      \033[0m 0;30;47m               \033[0;37;41m Black      \033[0m 0;37;41m")
        print("\033[1;31;40m Bright Red     \033[0m 1;31;40m            \033[0;31;47m Red        \033[0m 0;31;47m               \033[0;37;42m Black      \033[0m 0;37;42m")
        print("\033[1;32;40m Bright Green   \033[0m 1;32;40m            \033[0;32;47m Green      \033[0m 0;32;47m               \033[0;37;43m Black      \033[0m 0;37;43m")
        print("\033[1;33;40m Yellow         \033[0m 1;33;40m            \033[0;33;47m Brown      \033[0m 0;33;47m               \033[0;37;44m Black      \033[0m 0;37;44m")
        print("\033[1;34;40m Bright Blue    \033[0m 1;34;40m            \033[0;34;47m Blue       \033[0m 0;34;47m               \033[0;37;45m Black      \033[0m 0;37;45m")
        print( "\033[1;35;40m Bright Magenta \033[0m 1;35;40m            \033[0;35;47m Magenta    \033[0m 0;35;47m               \033[0;37;46m Black      \033[0m 0;37;46m")
        print("\033[1;36;40m Bright Cyan    \033[0m 1;36;40m            \033[0;36;47m Cyan       \033[0m 0;36;47m               \033[0;37;47m Black      \033[0m 0;37;47m")
        print("\033[1;37;40m White          \033[0m 1;37;40m            \033[0;37;40m Light Grey \033[0m 0;37;40m               \033[0;37;48m Black      \033[0m 0;37;48m")


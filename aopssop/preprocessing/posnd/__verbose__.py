from singleton.singleton import Singleton
from termcolor import colored


@Singleton
class PrintLog:

    def __init__(self):
        self.mode = False
        self.severity = 0

    @staticmethod
    def __array_to_int__(a):
        return "".join(list(map(lambda v: str(v), a)))

    def set_print_mode(self, mode):
        self.mode = mode

    def set_severity_level(self, severity):
        self.severity = {"status": 0, "info": 1, "warning": 2}[severity] \
            if severity in ["status", "info", "warning"] else 999

    def status(self, msg):
        if self.mode and self.severity < 1:
            if type(msg) != str:
                msg = self.__array_to_int__(msg)
            print(colored("     " + msg, 'white'))

    def info(self, msg):
        if self.mode and self.severity < 2:
            if type(msg) != str:
                msg = self.__array_to_int__(msg)
            print(colored(msg, 'yellow'))

    def warn(self, msg):
        if self.mode and self.severity < 3:
            if type(msg) != str:
                msg = self.__array_to_int__(msg)
            print(colored(msg, 'red'))

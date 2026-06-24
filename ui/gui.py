# -*- coding: utf-8 -*-

"""GUI 入口"""

import sys
from PyQt5.QtWidgets import QApplication
from smart_framing.ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
from cx_Freeze import setup, Executable

setup(
    name="picmatch",
    version="0.1",
    description="My GUI application!",
    executables=[Executable("SIFT.py")],
)
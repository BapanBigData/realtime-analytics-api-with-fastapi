from setuptools import setup, find_packages

setup(
    name="live_race_intel",
    version="0.1.0",
    description="Modular ML + FastAPI project",
    author="Bapan Bairagya",
    packages=find_packages(include=["src", "src.*", "app", "app.*"]),
    package_dir={
        "src": "src",
        "app": "app"
    },
    python_requires="==3.8",
)

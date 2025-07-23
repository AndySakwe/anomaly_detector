from setuptools import setup, find_packages

setup(
    name="anomaly-detector",
    version="1.0.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "joblib"
    ],
    entry_points={
        'console_scripts': [
            'anomalydetect=anomaly_detector.predictor:main'
        ]
    },
    author="AndySakwe",
    description="CLI tool for anomaly detection in network traffic",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

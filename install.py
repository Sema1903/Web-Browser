import subprocess
import sys

def install_requirements():
    """Установка всех зависимостей"""
    requirements = [
        'PyQt5>=5.15.0',
        'PyQtWebEngine>=5.15.0',
        'requests>=2.25.0',
        'beautifulsoup4>=4.9.0',
        'nltk>=3.5',
        'scikit-learn>=0.24.0',
        'numpy>=1.19.0',
    ]
    
    print("Установка зависимостей...")
    
    for package in requirements:
        print(f"Установка {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    print("\nУстановка данных NLTK...")
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    
    print("\n✅ Все зависимости успешно установлены!")
    print("\nЗапуск приложения:")
    print("python search_engine.py")

if __name__ == '__main__':
    install_requirements()
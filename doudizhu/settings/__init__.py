import importlib
import os

TORNADO_SETTINGS_MODULE = os.getenv('TORNADO_SETTINGS_MODULE', 'settings.dev')
print("Settings: ", TORNADO_SETTINGS_MODULE)
settings = importlib.import_module(TORNADO_SETTINGS_MODULE)

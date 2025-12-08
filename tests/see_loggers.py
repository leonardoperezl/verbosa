import logging

# ... (Import your libraries here, e.g., import requests) ...

# Get the dictionary of all loggers
loggers = [name for name in logging.root.manager.loggerDict]

print(f"Found {len(loggers)} loggers:")
for name in sorted(loggers):
    print(f" - {name}")
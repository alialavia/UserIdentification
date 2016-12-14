import importlib

req_type = "Identification"
req_module = getattr(importlib.import_module("src.request"), req_type)
req = getattr(req_module, "blabla")

print req_module
req()

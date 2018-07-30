import requests


r = requests.get('http://127.0.0.1:1994/get_aid')
print(r.json())

r = requests.get('http://127.0.0.1:1994/ts_ready/5')
print(r.json())


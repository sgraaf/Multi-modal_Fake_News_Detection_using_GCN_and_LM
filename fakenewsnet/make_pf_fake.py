import os
import json
import csv
import glob

DATA_DIR = os.getcwd() + "/PolitiFactFakeNewsContent/*.json"
all_fakes_dirs = glob.glob(DATA_DIR)
with open('../PolitiFact_fake_news_content.csv', 'w') as csv_file:
	csv_writer = csv.writer(csv_file, delimiter=',')
	csv_writer.writerow(['id', 'title', 'text'])
	for i in all_fakes_dirs:
		with open(i, 'r') as f:
			data = json.load(f)
			idx = i.split('PolitiFact_')[1].replace('.json','')
			csv_writer.writerow([idx, data['title'],data['text']])
			

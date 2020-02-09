import yaml
import os 
import csv 

home_dir=''
o_dir=os.path.join(home_dir,'Annotations_RGB')
c=0
with open ('labels.csv',mode='w') as w_file:
	csv_writer=csv.writer(w_file,delimiter=',')
	csv_writer.writerow(['f_name','label','size_x','size_y','x_max','y_max','y_min'])

	for file in os.listdir(o_dir):
		#print(file)
		directory=os.path.join(o_dir,file)
		with open(directory, 'r') as stream:
			doc=(yaml.load(stream))
			try:
				for item,d in doc.items():
					#print(doc['annotation']['object'])
					#print(len(doc['annotation']['object']))
					try:
						for i in range (int(len(doc['annotation']['object']))):	
							csv_writer.writerow([doc['annotation']['filename'],doc['annotation']['object'][i]['name'],doc['annotation']['size']['height'],doc['annotation']['size']['width'],doc['annotation']['object'][i]['bndbox']['xmax'],doc['annotation']['object'][i]['bndbox']['xmin'],doc['annotation']['object'][i]['bndbox']['ymax'],doc['annotation']['object'][i]['bndbox']['ymin']])
					except:
						csv_writer.writerow([doc['annotation']['filename'],'NA',doc['annotation']['size']['height'],doc['annotation']['size']['width'],0,0,0,0])

			except yaml.YAMLError as exc:
				print(exc)
	print(c)
	c+=1
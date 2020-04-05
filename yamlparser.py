import yaml
import os 
import csv 
import json 
data={}
l=[]
home_dir=''
o_dir=os.path.join(home_dir,'Annotations_RGB')
c=0
with open ('labels.csv',mode='w') as w_file:
	csv_writer=csv.writer(w_file,delimiter=',')
	csv_writer.writerow(['f_name','label','size_x','size_y','x_max','x_min','y_max','y_min'])

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
							d={}
							d['f_name']=doc['annotation']['filename']
							d['label']=doc['annotation']['object'][i]['name']
							d['size_x']=int(doc['annotation']['size']['height'])
							d['size_y']=int(doc['annotation']['size']['width'])
							d['dim']=[int(doc['annotation']['object'][i]['bndbox']['xmax']),int(doc['annotation']['object'][i]['bndbox']['xmin']),int(doc['annotation']['object'][i]['bndbox']['ymax']),int(doc['annotation']['object'][i]['bndbox']['ymin'])]
							l.append(d)
							csv_writer.writerow([doc['annotation']['filename'],doc['annotation']['object'][i]['name'],doc['annotation']['size']['height'],doc['annotation']['size']['width'],doc['annotation']['object'][i]['bndbox']['xmax'],doc['annotation']['object'][i]['bndbox']['xmin'],doc['annotation']['object'][i]['bndbox']['ymax'],doc['annotation']['object'][i]['bndbox']['ymin']])
					except:
						d={}
						d['f_name']=doc['annotation']['filename']
						d['label']='NA'
						d['size_x']=int(doc['annotation']['size']['height'])
						d['size_y']=int(doc['annotation']['size']['width'])
						d['dim']=[]
						l.append(d)	
						csv_writer.writerow([doc['annotation']['filename'],'NA',doc['annotation']['size']['height'],doc['annotation']['size']['width'],0,0,0,0])

			except yaml.YAMLError as exc:
				print(exc)
	#print(c)
	c=c+1
print(c)
data['obj']=l
with open('Mobility.json', 'w') as outfile:
    json.dump(data, outfile)